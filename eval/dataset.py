import json
import os

import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset

from utils import *


class VNQADataset(Dataset):
    """
    Initialize a dataset by providing the data directory, the list of filenames corresponding to the
    examples and the labels that can be indexed by the filenames.
    """
    def __init__(self,
                 q_dir,
                 v_dir,
                 filenames,
                 labels,
                 q_only=False,
                 v_only=False,
                 max_q_len=MAX_Q_LEN,
                 num_classes=NUM_CLASSES,
                 q_metadata=False):
        assert not (q_only and v_only), "Can't have both question- and video-only modes!"
        self.q_only = q_only
        self.v_only = v_only
        self.num_classes = num_classes
        self.max_q_len = max_q_len

        assert os.path.exists(q_dir), "Non-existent question directory!"
        assert os.path.exists(v_dir), "Non-existent video directory!"
        self.q_dir = q_dir
        self.v_dir = v_dir
        self.filenames = np.array(filenames)
        self.labels = labels
        self.new_frame_dropping = True

        self.q_metadata = q_metadata
        if self.q_metadata:
            self.q_ids = json.load(open(RAW_QUESTIONS_FILE, 'r'))


    """
    The total number of examples in the dataset.
    """
    def __len__(self):
        return self.filenames.shape[0]


    """
    Loads and returns the dataset example specified by the given index.
    """
    def __getitem__(self, index):
        filename = self.filenames[index]

        X = {}
        if not self.q_only:
            # Prepare container for video
            X_vid = np.empty(shape=(3, VID_HEIGHT, VID_WIDTH, MAX_NUM_VIDEO_FRAMES))

            # Read video frames
            vid = cv2.VideoCapture(os.path.join(self.v_dir, filename + '.mp4'))
            count = 0
            while True:
                ok, image = vid.read()
                if not ok:
                    break
                X_vid[:, :, :, count] = image.transpose(2, 0, 1)
                count += 1
            X_vid = X_vid[:, :, :, :count]

            vid.release()
            cv2.destroyAllWindows()

            # Subsample video
            vid_len = count
            X_vid_final = np.zeros(
                shape=(3, VID_HEIGHT, VID_WIDTH, MAX_ALLOWED_NUM_FRAMES_DROPPING))
            count = 0
            for i in range(0, vid_len, DROP_EVERY_N_FRAMES):
                idx_keep = random.randint(i, min(i + DROP_EVERY_N_FRAMES, vid_len) - 1)
                X_vid_final[:, :, :, count] = X_vid[:, :, :, idx_keep]
                count += 1
            X_vid = torch.from_numpy(X_vid_final)
            vid_len = count

            X['video'] = X_vid / 255.0
            X['v_len'] = vid_len

        if not self.v_only:
            X_q = np.load(os.path.join(self.q_dir, filename + '.npy'))
            X_q = torch.from_numpy(X_q)
            q = torch.LongTensor(np.zeros((self.max_q_len, )))
            q[:X_q.shape[0]] = X_q
            X['question'] = q
            X['q_len'] = X_q.shape[0]

        if self.q_metadata:
            X['q_id'] = self.q_ids[filename]

        y = self.labels[filename]
        return X, y


    """
    Provides inverse class weights for the cross-entropy loss function.
    """
    def get_class_weights(self):
        len_dataset = len(self.filenames)
        classes = np.array([self.labels[filename] for filename in self.filenames])

        class_weights = np.array(
            [(1.0 / float((classes == i).sum())) for i in range(self.num_classes)]
        )

        return class_weights
