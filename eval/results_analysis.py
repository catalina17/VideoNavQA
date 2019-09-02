import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--q_category', type=str, choices=['equals_attr', 'count', 'compare_count',
                                                       'compare_size', 'exist', 'query_color',
                                                       'query_obj', 'query_room', 'all'])

args = parser.parse_args()

yt = np.load('t_' + args.checkpoint_path + '.npy')
yp = np.load('p_' + args.checkpoint_path + '.npy')
q_ids = np.load('q_' + args.checkpoint_path + '.npy')

q_categories = {
    'equals_attr': [0, 1, 2, 3, 4, 5, 13],
    'count': [8, 9, 10, 11],
    'compare_count': [6, 7],
    'compare_size': [12, 14],
    'exist': [15, 16, 17, 18, 19, 20],
    'query_color': [21, 22],
    'query_obj': [23, 24],
    'query_room': [25, 26, 27],
}

for q_cat in q_categories:
    if not args.q_category == 'all' and q_cat != args.q_category:
        continue

    print('>>> Stats for %s:' % q_cat)
    all_answers_for_category_count = 0
    hit_answers_for_category_count = 0
    for q_id in q_categories[q_cat]:
        targets = yt[np.where(q_ids == q_id)[0]]
        if len(targets) == 0:
            continue

        predictions = yp[np.where(q_ids == q_id)[0]]
        hit = (predictions == targets).sum()

        print('Accuracy for question type %d: %.4f (%d\%d)' % (q_id,
                                                               100 * float(hit) / len(targets),
                                                               hit, len(targets)))
        all_answers_for_category_count += len(targets)
        hit_answers_for_category_count += hit

    print('Accuracy for question category %s: %.4f (%d\%d)' %\
          (q_cat, 100 * float(hit_answers_for_category_count) / all_answers_for_category_count,
           hit_answers_for_category_count, all_answers_for_category_count))
