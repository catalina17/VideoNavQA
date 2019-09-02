import argparse
import json

import numpy as np
import torch
import torch.nn as nn
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
parser.add_argument('--l_rate', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--stats_after_every', type=int, default=50)
parser.add_argument('--use_class_weights', type=bool, default=True)

# Other args
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--labels_file', type=str, default='../data/labels.json')
parser.add_argument('--split_file', type=str, default='../data/split.json')
parser.add_argument('--q_dir', type=str, default='../data/encoded_questions/')
parser.add_argument('--v_dir', type=str, default='../data/videos/')

args = parser.parse_args()


def save_checkpoint(state):
    torch.save(state, args.checkpoint_path)


def train_epoch(epoch, model, data_loader, optimizer, loss_fn, batch_size):
    model.train()

    # Values to be monitored
    avg_loss = 0
    hit = 0
    y_pred = np.array([])
    y_target = np.array([])

    num_examples = 0
    for i, data in enumerate(data_loader, 0):
        Xs, ys = data
        if len(ys) < batch_size:
            continue
        num_examples += len(ys)

        q_inputs = Variable(Xs['question'])
        q_lens = Variable(Xs['q_len'])
        ys = Variable(ys)
        # CUDA inputs
        if use_cuda:
            q_inputs, q_lens, ys = q_inputs.cuda(), q_lens.cuda(), ys.cuda()

        # Sort questions by descending length (number of tokens)
        init_lens = q_lens
        q_lens, perm_idx = q_lens.sort(0, descending=True)
        q_inputs, ys = q_inputs[perm_idx], ys[perm_idx]
        y_target = np.append(y_target, ys.cpu().numpy())

        if args.model == 'lstm':
            optimizer.zero_grad()
            model.init_hidden()
        output = model(q_inputs, q_lens)

        # Loss
        loss = loss_fn(output, ys)
        avg_loss += loss
        # Count hit examples
        pred_class = output.data.max(1)[1]
        y_pred = np.append(y_pred, (pred_class.cpu()).numpy())
        hit += (pred_class == ys.data).sum()

        loss.backward()
        optimizer.step()

    if epoch % args.stats_after_every == 0:
        # Compute metrics
        f1_w = f1_score(y_target, y_pred, average='weighted')
        f1_micro = f1_score(y_target, y_pred, average='micro')
        print('Train Epoch: {}\tAverage loss: {:.6f}\tF1: w{:.4f}, micro{:.4f}'.format(
            epoch, f1_w, f1_micro))


def val_epoch(model, data_loader, loss_fn, num_classes, batch_size):
    model.eval()

    # Values to be monitored
    test_loss = 0
    hit = 0
    y_pred = np.array([])
    y_target = np.array([])

    num_examples = 0
    for i, data in enumerate(data_loader, 0):
        Xs, ys = data
        if len(ys) < batch_size:
           continue
        num_examples += len(ys)

        q_inputs = Variable(Xs['question'])
        q_lens = Variable(Xs['q_len'])
        ys = Variable(ys)
        # CUDA inputs
        if use_cuda:
            q_inputs, q_lens, ys = q_inputs.cuda(), q_lens.cuda(), ys.cuda()

        # Sort questions by descending length (number of tokens)
        init_lens = q_lens
        q_lens, perm_idx = q_lens.sort(0, descending=True)
        q_inputs, ys = q_inputs[perm_idx], ys[perm_idx]
        y_target = np.append(y_target, ys.cpu().numpy())

        output = model(q_inputs, q_lens)

        # Loss
        test_loss += loss_fn(output, ys)
        # Process predictions
        pred_class = output.data.max(1)[1]
        y_pred = np.append(y_pred, (pred_class.cpu()).numpy())
        # Count hit examples per class
        hit += (pred_class == ys.data).sum()

    # Compute metrics
    f1_w = f1_score(y_target, y_pred, average='weighted')
    f1_micro = f1_score(y_target, y_pred, average='micro')
    print('Validation:\tAverage loss: {:.6f}, F1: w{:.4f}, micro{:.4f}'.format(
        test_loss.item() / num_examples, f1_w, f1_micro))

    torch.cuda.empty_cache()

    return f1_micro


if __name__=='__main__':

    with open(args.split_file, 'r') as f:
        split = json.load(f)
    train_file_ids = split['train']
    test_file_ids = split['val']
    print('%d train examples, %d validation examples' % (len(train_file_ids), len(test_file_ids)))

    with open(args.labels_file, 'r') as f:
        labels = json.load(f)
    num_classes = args.num_classes

    # Initialize datasets for training and testing
    train_data = VNQADataset(q_dir=args.q_dir, v_dir=args.v_dir, q_only=True,
                             filenames=train_file_ids, labels=labels, num_classes=num_classes)
    test_data = VNQADataset(q_dir=args.q_dir, v_dir=args.v_dir, q_only=True,
                            filenames=test_file_ids, labels=labels, num_classes=num_classes)

    # Create DataLoader objects for training and test datasets
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
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

    optimizer = optim.Adam(model.parameters(), lr=args.l_rate)
    best_acc = 0
    for epoch in range(args.num_epochs):
        train_epoch(epoch + 1, model, train_loader, optimizer, loss_fn, args.batch_size)
        if (epoch + 1) % args.stats_after_every == 0:
            acc = test(model, test_loader, loss_fn, num_classes, args.batch_size)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint({
                    'epoch': epoch,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'val_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                })
