import numpy as np
import torch

from models.obj_detector import ObjDetectCNN

BASE_DIR = '../data/'

# Dir paths
QUESTIONS_DIR = BASE_DIR + 'encoded_questions'
VIDEOS_DIR = BASE_DIR + 'videos'

# File paths
LABELS_FILE = BASE_DIR + 'labels.json'
OBJ_DETECTOR_PATH = BASE_DIR + 'obj_detect.pt'
RAW_QUESTIONS_FILE = BASE_DIR + 'q_ids.json'
SPLIT_FILE = BASE_DIR + 'split.json'

# Numeric constants
DROP_EVERY_N_FRAMES = 4
MAX_ALLOWED_NUM_FRAMES_DROPPING = 35
MAX_NUM_VIDEO_FRAMES = 400
MAX_Q_LEN = 56
NUM_CLASSES = 70
VID_HEIGHT = 160
VID_WIDTH = 208

use_cuda = torch.cuda.is_available()


def per_class_accuracies(y_target, y_pred, num_classes):
    accs = []
    for i in range(num_classes):
        total = np.where(y_target == i)[0].size
        idxs = np.where(y_target == i)[0]
        hits = np.where(y_pred[idxs] == i)[0].size
        res = (float(hits) / float(total)) if total != 0 else 0.0
        accs.append(res)

    return np.array(accs)


def get_object_detector():
    model = ObjDetectCNN(nb_classes=27,
                         num_filters=512,
                         tail_hidden_dim=1024,
                         tail_dropout_p=0,
                         logits=True,
                         pretrained_features=True)
    model.load_state_dict(torch.load(OBJ_DETECTOR_PATH)['state_dict'])
    model.eval()
    return model
