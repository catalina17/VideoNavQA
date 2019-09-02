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
from models.q_only_lstm import QOnlyLSTM
from models.q_only_bow import QOnlyBOW

from utils import *


parser = argparse.ArgumentParser()

# Model args
parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--model', type=str, choices=['lstm', 'bow'])
parser.add_argument('--num_classes', type=int, default=70)
parser.add_argument('--vocab_size', type=int, default=134)

# Optimization args
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--use_class_weights', type=bool, default=True)

# Other args
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--labels_file', type=str, default='../data/labels.json')
parser.add_argument('--split_file', type=str, default='../data/split.json')
parser.add_argument('--q_dir', type=str, default='../data/encoded_questions/')
parser.add_argument('--v_dir', type=str, default='../data/videos/')

args = parser.parse_args()


def test(model, data_loader, loss_fn, num_classes, batch_size):
    model.eval()

    # Values to be monitored
    test_loss = 0
    hit = 0
    y_pred = np.array([])
    y_target = np.array([])
    qs = np.array([])

    num_examples = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            Xs, ys = data
            num_examples += len(ys)
            num_real_examples = ys.size(0)

            if len(ys) < batch_size:
                padded = batch_size - ys.size(0)
                Xs['question'] = F.pad(Xs['question'], (0, 0, 0, padded), 'constant', 0)
                Xs['q_len'] = F.pad(Xs['q_len'], (0, padded), 'constant', 1)
                Xs['q_id'] = F.pad(Xs['q_id'], (0, padded), 'constant', 35)
                ys = F.pad(ys, (0, padded), 'constant', 0)

            q_inputs = Variable(Xs['question'])
            q_lens = Variable(Xs['q_len'])
            q_ids = Variable(Xs['q_id'])
            ys = Variable(ys)
            # CUDA inputs
            if use_cuda:
                q_inputs, q_lens, ys = q_inputs.cuda(), q_lens.cuda(), ys.cuda()

            # Sort questions by descending length (number of tokens)
            init_lens = q_lens
            q_lens, perm_idx = q_lens.sort(0, descending=True)
            q_inputs, ys, q_ids = q_inputs[perm_idx], ys[perm_idx], q_ids[perm_idx]
            y_target = np.append(y_target, ys[:num_real_examples].cpu().numpy())
            qs = np.append(qs, q_ids[:num_real_examples].cpu().numpy())

            output = model(q_inputs, q_lens)[:num_real_examples]
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
    print('Testing:\tAverage loss: {:.6f}, F1: w{:.4f}, micro{:.4f}'.format(
        test_loss.item() / num_examples, f1_w, f1_micro))

    return y_target, y_pred, qs


if __name__=='__main__':

    with open(args.split_file, 'r') as f:
        split = json.load(f)
    train_file_ids = split['train']
    test_file_ids = split['test']
    print('%d test examples' % len(test_file_ids))

    with open(args.labels_file, 'r') as f:
        labels = json.load(f)
    num_classes = args.num_classes

    # Initialize datasets
    train_data = VNQADataset(q_dir=args.q_dir, v_dir=args.v_dir, q_only=True,
                             filenames=train_file_ids, labels=labels, num_classes=num_classes)
    test_data = VNQADataset(q_dir=args.q_dir, v_dir=args.v_dir, q_only=True,
                            filenames=test_file_ids, labels=labels, num_classes=num_classes)

    # Create DataLoader objects
    test_loader = DataLoader(dataset=test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)

    # Initialize the model
    if args.model == 'lstm':
        model = QOnlyLSTM(batch_size=args.batch_size,
                          embedding_size=args.embed_size,
                          hidden_size=args.hidden_size,
                          nb_classes=num_classes,
                          vocab_size=args.vocab_size)
    elif args.model == 'bow':
        model = QOnlyBOW(batch_size=args.batch_size,
                         embedding_size=args.embed_size,
                         nb_classes=num_classes,
                         vocab_size=args.vocab_size)

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
            print('=> No checkpoint existent! Aborting.')
            exit(-1)
        else:
            print('=> Restoring from checkpoint path %s' % args.checkpoint_path)
            checkpoint = torch.load(args.checkpoint_path)
            epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            acc = checkpoint['val_acc']
            print('==> Restored checkpoint from epoch %d (validation accuracy %.4f)' % (epoch, acc))

    # Save results (targets, predictions, question types)
    t, p, q = test(model, test_loader, loss_fn, num_classes, args.batch_size)
    np.save('t_' + args.checkpoint_path, t)
    np.save('p_' + args.checkpoint_path, p)
    np.save('q_' + args.checkpoint_path, q)
