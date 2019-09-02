import argparse
import json
import os

import numpy as np
import pprint as pp
import torch
import torch.nn as nn
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
parser.add_argument('--use_class_weights', type=bool, default=False)

# Optimization args
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--l_rate', type=float, default=5e-4)
parser.add_argument('--loss_reduction', type=str, choices=['sum', 'mean', 'elementwise_mean'])
parser.add_argument('--num_epochs', type=int, default=1)

# Other args
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--stats_after_every', type=int, default=10)
parser.add_argument('--val_only', type=bool, default=False)

args = parser.parse_args()


def save_checkpoint(state):
    torch.save(state, 'e' + str(state['epoch']) + '_' + args.checkpoint_path)


def train_epoch(epoch, model, data_loader, optimizer, loss_fn):
    model.train()

    # Values to be monitored
    avg_loss = 0
    hit = 0
    y_pred = np.array([])
    y_target = np.array([])

    num_examples = 0
    for i, data in enumerate(data_loader, 0):
        Xs, ys = data
        if len(ys) < args.batch_size:
            continue

        num_examples += len(ys)

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
        y_target = np.append(y_target, ys.cpu().numpy())

        optimizer.zero_grad()
        model.init_hidden()
        output = model(v_inputs, v_lens)

        # Loss
        loss = loss_fn(output, ys)
        avg_loss += loss
        # Count hit examples
        pred_class = output.data.max(1)[1]
        y_pred = np.append(y_pred, (pred_class.cpu()).numpy())
        hit += (pred_class == ys.data).sum()

        if (i + 1) % args.stats_after_every == 0:
            print('Average loss after %d iterations in epoch %d: %.6f' %
                (i + 1, epoch + 1, avg_loss.item() / num_examples))

        loss.backward()
        optimizer.step()

    # Compute F1 scores
    f1_w = f1_score(y_target, y_pred, average='weighted')
    f1_micro = f1_score(y_target, y_pred, average='micro')
    print('Train Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {}/{}\tF1: w{:.4f}, micro{:.4f}\n'.format(
          epoch, avg_loss.item() / num_examples, hit, num_examples, f1_w, f1_micro))

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'train_f1w': f1_w,
        'train_f1uw': f1_uw,
        'train_f1micro': f1_micro,
        'optimizer': optimizer.state_dict(),
    })

    torch.cuda.empty_cache()


def val_epoch(model, data_loader, loss_fn, num_classes):
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
		y_target = np.append(y_target, ys.cpu().numpy())

		model.init_hidden()
		output = model(v_inputs, v_lens)

		# Loss
		val_loss += loss_fn(output, ys)

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
    print('Validation:\tAverage loss: {:.6f}, Accuracy: {}/{}, F1: w{:.4f}, micro{:.4f}\n'.format(
          val_loss.item() / num_examples, hit, num_examples, f1_w, f1_micro))

    torch.cuda.empty_cache()


if __name__=='__main__':

    with open(SPLIT_FILE, 'r') as f:
        split = json.load(f)
    train_file_ids = split['train']
    val_file_ids = split['val']
    print('%d train examples, %d validation examples' % (len(train_file_ids), len(val_file_ids)))

    with open(LABELS_FILE, 'r') as f:
        labels = json.load(f)

    # Initialize datasets for training and testing
    train_data = VNQADataset(q_dir=QUESTIONS_DIR, v_dir=VIDEOS_DIR, v_only=True,
                             filenames=train_file_ids, labels=labels)
    val_data = VNQADataset(q_dir=QUESTIONS_DIR, v_dir=VIDEOS_DIR, v_only=True,
                           filenames=val_file_ids, labels=labels)

    # Create DataLoader objects for training and test datasets
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            shuffle=True,
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
    optimizer = optim.Adam(model.parameters(), lr=args.l_rate)

    start_epoch = 0
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

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if not args.val_only:
            train_epoch(epoch, model, train_loader, optimizer, loss_fn)
        val_epoch(model, val_loader, loss_fn, num_classes)
