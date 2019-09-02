import argparse
import json
import os

import numpy as np
import pprint as pp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import VNQADataset
from models.v_only_cnn2d_lstm import VideoOnlyCNN2DLSTM

from utils import *


parser = argparse.ArgumentParser()

# Model args
parser.add_argument('--num_classes', type=int, default=70)

# Optimization args
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--loss_reduction', type=str, choices=['sum', 'mean', 'elementwise_mean'])
parser.add_argument('--use_class_weights', type=bool, default=False)

# Other args
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()


def test(model, data_loader, loss_fn, num_classes):
    model.eval()

    # Values to be monitored
    test_loss = 0
    hit = 0
    y_pred = np.array([])
    y_target = np.array([])

    num_examples = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            Xs, ys = data
            num_examples += len(ys)
            num_real_examples = ys.size(0)
            if len(ys) < args.batch_size:
                padded = args.batch_size - ys.size(0)
                Xs['video'] = F.pad(Xs['video'], (0, 0, 0, 0, 0, 0, 0, 0, 0, padded), 'constant', 0)
                Xs['v_len'] = F.pad(Xs['v_len'], (0, padded), 'constant', 1)
                ys = F.pad(ys, (0, padded), 'constant', 0)

            v_inputs = Variable(Xs['video']).float()
            v_lens = Variable(Xs['v_len'])
            ys = Variable(ys)
            # CUDA inputs
            if use_cuda:
                v_inputs, v_lens, ys = v_inputs.cuda(), v_lens.cuda(), ys.cuda()

            # Sort video examples by descending length (number of frames)
            init_lens = v_lens
            v_lens, perm_idx = v_lens.sort(0, descending=True)
            v_inputs, ys = v_inputs[perm_idx], ys[perm_idx]
            y_target = np.append(y_target, ys[:num_real_examples].cpu().numpy())

            model.init_hidden()
            output = model(v_inputs, v_lens)[:num_real_examples]
            ys = ys[:num_real_examples]

            # Loss
            test_loss += loss_fn(output, ys)

            # Process predictions
            pred_class = output.data.max(1)[1]
            y_pred = np.append(y_pred, (pred_class.cpu()).numpy())

            # Count hit examples per class
            hit += (pred_class == ys.data).sum()

    accs = per_class_accuracies(y_target, y_pred, num_classes)
    pp.pprint({i : accs[i] for i in np.nonzero(accs)[0].tolist()})
    # Compute F1 scores
    f1_w = f1_score(y_target, y_pred, average='weighted')
    f1_micro = f1_score(y_target, y_pred, average='micro')
    print('Testing:\tAverage loss: {:.6f}, Accuracy: {}/{}, F1: w{:.4f}, micro{:.4f}\n'.format(
          test_loss.item() / num_examples, hit, num_examples, f1_w, f1_micro))


if __name__=='__main__':

    with open(SPLIT_FILE, 'r') as f:
        split = json.load(f)
    train_file_ids = split['train']
    test_file_ids = split['test']
    print('%d train examples, %d test examples' % (len(train_file_ids), len(test_file_ids)))

    with open(LABELS_FILE, 'r') as f:
        labels = json.load(f)

    # Initialize datasets for training and testing
    train_data = VNQADataset(q_dir=QUESTIONS_DIR, v_dir=VIDEOS_DIR, v_only=True,
                             filenames=train_file_ids, labels=labels)
    test_data = VNQADataset(q_dir=QUESTIONS_DIR, v_dir=VIDEOS_DIR, v_only=True,
                            filenames=test_file_ids, labels=labels)

    # Create DataLoader objects for training and test datasets
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers)

    # Initialize the model
    num_classes = args.num_classes
    model = VideoOnlyCNN2DLSTM(batch_size=args.batch_size, nb_classes=num_classes)

    # Loss function
    class_weights = None
    if args.use_class_weights:
        class_weights = torch.FloatTensor(train_data.get_class_weights()).cuda()
        print('Using class weights', class_weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    if use_cuda:
        print('Using CUDA')
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    print(model)
    # Load checkpoint
    if not args.checkpoint_path is None:
        if not os.path.exists(args.checkpoint_path):
            print('=> No checkpoint existent - will save the model here')
        else:
            print('=> Restoring from checkpoint path %s' % args.checkpoint_path)
            checkpoint = torch.load(args.checkpoint_path)
            epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            if 'val_acc' in checkpoint:
                acc = checkpoint['val_acc']
            else:
                acc = -1.0
            print('==> Restored checkpoint from epoch %d (validation accuracy %.4f)' % (epoch, acc))

    test(model, test_loader, loss_fn, num_classes)
