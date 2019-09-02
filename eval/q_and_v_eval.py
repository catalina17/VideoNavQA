import argparse
import json
import os

import numpy as np
import pprint as pp
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import torch.optim as optim

from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import VNQADataset
from demo import get_frcnn_feature_extractor

from models.film_attn_pt_stem import FiLMAttnPretrainedStem
from models.film_global_pooling_pt_stem import FiLMGlobalPoolingPretrainedStem
from models.mac import MACNetwork
from models.q_concat_cnn2d_lstm import QConcatCNN2DLSTM
from models.q_concat_cnn3d import QConcatCNN3D
from models.time_multi_hop_pt_stem import TimeMultiHopFiLMPretrainedStem

from utils import *


parser = argparse.ArgumentParser()

# Model args
parser.add_argument('--model', type=str, choices=['concat2d', 'concat3d', 'film_gp_pt',
                                                  'film_attn_pt', 'mac', 'time_multi_hop'])
parser.add_argument('--num_classes', type=int, default=70)
parser.add_argument('--q_encoder', type=str, choices=['lstm', 'bow'], default='lstm')
parser.add_argument('--use_obj_detector', type=bool, default=True)
parser.add_argument('--use_visual_features', type=bool, default=True)
parser.add_argument('--vocab_size', type=int, default=134)

# Model hyperparameters
parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--at_hidden_size', type=int, default=128)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_res_block_channels', type=int, default=512)
parser.add_argument('--num_input_channels', type=int, default=512)
parser.add_argument('--num_tail_channels', type=int, default=16)
parser.add_argument('--mac_dim', type=int, default=512)
parser.add_argument('--mac_max_step', type=int, default=12)

# Optimization args
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--clip_value', type=float, default=1.0)
parser.add_argument('--l_rate', type=float, default=1e-4)
parser.add_argument('--loss_reduction', type=str, choices=['sum', 'mean', 'elementwise_mean'])
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--use_class_weights', type=bool, default=False)

# Other args
parser.add_argument('--best_acc', type=float, default=0)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--dataset', type=str, default='v3')
parser.add_argument('--frcnn_pretrained_path', type=str)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--stats_after_every', type=int, default=400)
parser.add_argument('--val_only', type=bool, default=False)

args = parser.parse_args()


def save_checkpoint(state, path):
    torch.save(state, path)


def train_epoch(epoch, model, data_loader, optimizer, loss_fn, feature_extractor, obj_detector):
    model.train()

    # Values to be monitored
    avg_loss = 0
    hit = 0
    y_pred = np.array([])
    y_target = np.array([])

    num_examples = 0
    optimizer.zero_grad()
    for i, data in enumerate(data_loader, 0):
        Xs, ys = data
        if len(ys) < args.batch_size:
            continue
        num_examples += len(ys)

        q_inputs = Variable(Xs['question'])
        q_lens = Variable(Xs['q_len'])
        v_inputs = Variable(Xs['video']).float()
        v_lens = Variable(Xs['v_len'])
        ys = Variable(ys)

        # CUDA inputs
        if use_cuda:
            q_inputs, q_lens, v_inputs, v_lens, ys =\
                q_inputs.cuda(), q_lens.cuda(), v_inputs.cuda(), v_lens.cuda(), ys.cuda()

        # Get visual features
        if args.use_visual_features:
            v_features = []
            with torch.no_grad():
                for j in range(v_inputs.shape[-1]):
                    features = feature_extractor(v_inputs[:, :, :, :, j])
                    if args.use_obj_detector:
                        features = obj_detector(features)
                    v_features.append(features)
            v_inputs = torch.stack(v_features).permute(1, 2, 3, 4, 0)

        # Sort examples by descending video length (number of frames)
        init_lens = v_lens
        v_lens, perm_idx = v_lens.sort(0, descending=True)
        v_inputs, q_inputs, q_lens, ys =\
            v_inputs[perm_idx], q_inputs[perm_idx], q_lens[perm_idx], ys[perm_idx]
        y_target = np.append(y_target, ys.cpu().numpy())

        if args.model != 'mac':
            model.init_hidden()
        output = model(v_inputs, q_inputs, v_lens, q_lens)

        # Loss
        loss = loss_fn(output, ys)
        avg_loss += loss
        # Process predictions
        pred_class = output.data.max(1)[1]
        y_pred = np.append(y_pred, (pred_class.cpu()).numpy())
        # Count hit examples
        hit += (pred_class == ys.data).sum()

        if (i + 1) % args.stats_after_every == 0:
            print('Average loss after %d iterations in epoch %d: %.6f' %
                (i + 1, epoch + 1, avg_loss.item() / num_examples))

        loss.backward()
        clip_grad_norm(model.parameters(), args.clip_value)
        optimizer.step()
        optimizer.zero_grad()

    # Compute metrics
    f1_w = f1_score(y_target, y_pred, average='weighted')
    f1_micro = f1_score(y_target, y_pred, average='micro')
    print(
        'Train Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {}/{}\tF1: w{:.4f}, micro{:.4f}\n'.format(
            epoch, avg_loss.item() / num_examples, hit, num_examples, f1_w, f1_micro))

    state_dict = model.state_dict()
    save_checkpoint({
        'epoch': epoch,
        'model': args.model,
        'state_dict': state_dict,
        'train_f1w': f1_w,
        'train_f1micro': f1_micro,
        'optimizer': optimizer.state_dict(),
    }, 'e' + str(epoch) + '_' + args.checkpoint_path)


def val_epoch(model, data_loader, loss_fn, num_classes, feature_extractor, obj_detector):
    model.eval()

    # Values to be monitored
    val_loss = 0
    hit = 0
    y_pred = np.array([])
    y_target = np.array([])

    num_examples = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            Xs, ys = data
            if len(ys) < args.batch_size:
               continue
            num_examples += len(ys)

            q_inputs = Variable(Xs['question'])
            q_lens = Variable(Xs['q_len'])
            v_inputs = Variable(Xs['video']).float()
            v_lens = Variable(Xs['v_len'])
            ys = Variable(ys)

            # CUDA inputs
            if use_cuda:
                q_inputs, q_lens, v_inputs, v_lens, ys =\
                    q_inputs.cuda(), q_lens.cuda(), v_inputs.cuda(), v_lens.cuda(), ys.cuda()

            # Get visual features
            if args.use_visual_features:
                v_features = []
                for j in range(v_inputs.shape[-1]):
                    features = feature_extractor(v_inputs[:, :, :, :, j])
                    if args.use_obj_detector:
                        features = obj_detector(features)
                    v_features.append(features)
                v_inputs = torch.stack(v_features).permute(1, 2, 3, 4, 0)

            # Sort examples by descending video length (number of frames)
            init_lens = v_lens
            v_lens, perm_idx = v_lens.sort(0, descending=True)
            v_inputs, q_inputs, q_lens, ys =\
                v_inputs[perm_idx], q_inputs[perm_idx], q_lens[perm_idx], ys[perm_idx]
            y_target = np.append(y_target, ys.cpu().numpy())

            if args.model != 'mac':
                model.init_hidden()
            output = model(v_inputs, q_inputs, v_lens, q_lens)

            # Loss
            val_loss += loss_fn(output, ys)
            # Process predictions
            pred_class = output.data.max(1)[1]
            y_pred = np.append(y_pred, (pred_class.cpu()).numpy())
            # Count hit examples per class
            hit += (pred_class == ys.data).sum()

    accs = per_class_accuracies(y_target, y_pred, num_classes)
    pp.pprint({i : accs[i] for i in np.nonzero(accs)[0].tolist()})

    # Compute metrics
    f1_w = f1_score(y_target, y_pred, average='weighted')
    f1_micro = f1_score(y_target, y_pred, average='micro')
    print('Validation:\tAverage loss: {:.6f}, Accuracy: {}/{}, F1: w{:.4f}, micro{:.4f}\n'.format(
        val_loss.item() / num_examples, hit, num_examples, f1_w, f1_micro))
    torch.cuda.empty_cache()


if __name__=='__main__':

    q_dir = QUESTIONS_DIR
    v_dir = VIDEOS_DIR
    labels_file = LABELS_FILE
    split_file = SPLIT_FILE

    with open(split_file, 'r') as f:
        split = json.load(f)
    train_file_ids = split['train']
    val_file_ids = split['val']

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    # Initialize datasets for training and validation
    train_data = VNQADataset(q_dir=q_dir, v_dir=v_dir, filenames=train_file_ids, labels=labels)
    val_data = VNQADataset(q_dir=q_dir, v_dir=v_dir, filenames=val_file_ids, labels=labels)
    print('%d train examples, %d validation examples' % (len(train_data), len(val_data)))

    # Create DataLoader objects for training and validation sets
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers)

    # Initialize the model
    num_classes = args.num_classes
    if args.model == 'concat2d':
        model = QConcatCNN2DLSTM(batch_size=args.batch_size,
                                 q_embedding_size=args.embed_size,
                                 nb_classes=num_classes,
                                 vocab_size=args.vocab_size)
    elif args.model == 'concat3d':
        model = QConcatCNN3D(batch_size=args.batch_size,
                             q_embedding_size=args.embed_size,
                             nb_classes=num_classes,
                             vocab_size=args.vocab_size)
   elif args.model == 'film_attn_pt':
       model = FiLMAttnPretrainedStem(batch_size=args.batch_size,
                                      q_embedding_size=args.embed_size,
                                      nb_classes=num_classes,
                                      q_encoder=args.q_encoder,
                                      num_input_channels=args.num_input_channels,
                                      num_res_block_channels=args.num_res_block_channels,
                                      num_res_blocks=args.num_res_blocks,
                                      hidden_size=args.hidden_size,
                                      at_hidden_size=args.at_hidden_size,
                                      max_num_frames=MAX_ALLOWED_NUM_FRAMES_DROPPING,
                                      vocab_size=args.vocab_size)
    elif args.model == 'film_gp_pt':
        model = FiLMGlobalPoolingPretrainedStem(batch_size=args.batch_size,
                                                q_embedding_size=args.embed_size,
                                                nb_classes=num_classes,
                                                num_input_channels=args.num_input_channels,
                                                num_res_block_channels=args.num_res_block_channels,
                                                num_res_blocks=args.num_res_blocks,
                                                hidden_size=args.hidden_size,
                                                num_tail_channels=args.num_tail_channels,
                                                q_encoder=args.q_encoder,
                                                vocab_size=args.vocab_size)
    elif args.model == 'mac':
        model = MACNetwork(n_vocab=args.vocab_size,
                           dim=args.mac_dim,
                           embed_hidden=args.embed_size,
                           max_step=args.mac_max_step,
                           classes=num_classes)
    elif args.model == 'time_multi_hop':
        model = TimeMultiHopFiLMPretrainedStem(batch_size=args.batch_size,
                                               q_embedding_size=args.embed_size,
                                               nb_classes=num_classes,
                                               num_input_channels=args.num_input_channels,
                                               num_res_block_channels=args.num_res_block_channels,
                                               num_res_blocks=args.num_res_blocks,
                                               num_tail_channels=args.num_tail_channels,
                                               hidden_size=args.hidden_size,
                                               vocab_size=args.vocab_size)

    # Load the pre-trained network for visual feature extraction
    feature_extractor = None
    if args.use_visual_features:
        feature_extractor = get_frcnn_feature_extractor(args.frcnn_pretrained_path)
        if not args.use_obj_detector:
            print(feature_extractor)
    obj_detector = None
    if args.use_obj_detector:
        obj_detector = get_object_detector()
        print(obj_detector)

    # Loss function
    class_weights = None
    if args.use_class_weights:
        class_weights = torch.FloatTensor(train_data.get_class_weights()).cuda()
        print('Using class weights', class_weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction=args.loss_reduction)

    if use_cuda:
        print('Using CUDA')
        model = model.cuda()
        if args.use_visual_features:
            feature_extractor = feature_extractor.cuda()
        loss_fn = loss_fn.cuda()
        if args.use_obj_detector:
            obj_detector = obj_detector.cuda()

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.l_rate)

    start_epoch = 0
    best_acc = args.best_acc
    # Load checkpoint
    if not args.checkpoint_path is None:
        if not os.path.exists(args.checkpoint_path):
            print('=> No checkpoint existent - will save the model here')
        else:
            print('=> Restoring from checkpoint path %s' % args.checkpoint_path)
            checkpoint = torch.load(args.checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('==> Restored checkpoint %s (epoch %d)' % (args.checkpoint_path, start_epoch))

    if args.model == 'mac':
        clip_value = 1.0
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # Train and validate the model
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if not args.val_only:
            train_epoch(
                epoch, model, train_loader, optimizer, loss_fn, feature_extractor, obj_detector)
        if args.model == 'mac':
            if epoch == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10.  # warmup
                print('learning rate %.5f' % optimizer.param_groups[0]['lr'])
            else:
                optimizer.param_groups[0]['lr'] = args.l_rate
                print('learning rate %.5f' % optimizer.param_groups[0]['lr'])
        val_epoch(model, val_loader, loss_fn, num_classes, feature_extractor, obj_detector)
