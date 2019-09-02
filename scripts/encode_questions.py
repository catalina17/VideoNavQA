import argparse
from collections import Counter
import json
import os
import re

import pprint as pp
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str)
parser.add_argument('--question_lengths_file', type=str)
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()


# Make a list of all unique tokens appearing in the questions
unique_tokens = []
filenames = os.listdir(args.data_dir)
for filename in filenames:
    if not filename.endswith('.json'):
        continue

    house_id = filename.split('.')[0]
    with open(os.path.join(args.data_dir, filename), 'r') as f:
        data = json.load(f)

    for traj_id in data:
        all_toks = [x.lower() for x in re.findall(r"[\w']+|/|\?", data[traj_id]['q_text'])]
        for tok in all_toks:
            if not tok in unique_tokens:
                unique_tokens.append(tok)

print(len(list(unique_tokens)), 'distinct tokens.')

q_lens = []
# Encode questions
for filename in filenames:
    if not filename.endswith('.json'):
        continue

    with open(os.path.join(args.data_dir, filename), 'r') as f:
        data = json.load(f)

    house_id = filename.split('.')[0]
    for traj_id in data:
        all_toks = [x.lower() for x in re.findall(r"[\w']+|/|\?", data[traj_id]['q_text'])]
        q_encoding = []
        for tok in all_toks:
            assert tok in unique_tokens, 'Token not in the list!'
            q_encoding.append(unique_tokens.index(tok) + 1)

        q_encoding = np.array(q_encoding)
        q_lens.append(q_encoding.shape[0])

        example_filename = '%s_%04d' % (house_id, int(traj_id))
        np.save(os.path.join(args.save_dir, example_filename + '.npy'), q_encoding)

q_lens = np.array(q_lens)
print('Question length distribution (mean/std/max):' q_lens.mean(), q_lens.std(), q_lens.max())
counter = Counter(q_lens)
for c in sorted(counter):
    print(c, counter[c])
np.save(args.question_lengths_file, q_lens)
