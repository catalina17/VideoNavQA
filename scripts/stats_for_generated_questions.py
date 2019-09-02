import argparse
import json
import os
import pprint as pp

from question_gen import QuestionGenerator

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str)

args = parser.parse_args()


def get_qs_for_house(hid):
    for file in files:
        if file.startswith(hid):
            f = open(os.path.join(args.data_dir, file), 'r')
            data = json.load(f)
            for traj_id in data:
                example = '%s_%04d' % (hid.split('.')[0], int(traj_id))

                template = data[traj_id]['q_id']
                if not template in qs_for_houses:
                    qs_for_houses[template] = {}

                template_qs_for_houses = qs_for_houses[template]
                key = data[traj_id]['q_text']
                if not key in template_qs_for_houses:
                    template_qs_for_houses[key] = { data[traj_id]['q_ans'] : 1 }
                elif not data[traj_id]['q_ans'] in template_qs_for_houses[key]:
                    ans = data[traj_id]['q_ans']
                    template_qs_for_houses[key][ans] = 1
                else:
                    ans = data[traj_id]['q_ans']
                    template_qs_for_houses[key][ans] += 1


def build_split():
    split = { 'train' : [], 'test' : [], 'val' : [] }
    splits = json.load(open('eqa_v1.json', 'r'))['splits']

    for file in files:
        f = open(os.path.join(args.data_dir, file), 'r')
        data = json.load(f)
        house_id = file.split('.')[0]

        for traj_id in data:
            example_id = '%s_%04d' % (house_id, int(traj_id))

            if house_id in splits['train']:
                split['train'].append(example_id)
            elif house_id in splits['test']:
                split['test'].append(example_id)
            elif house_id in splits['val']:
                split['val'].append(example_id)
            else:
                print('House id not found! ABORTING')
                exit(-1)

    print('Examples in each dataset split:')
    for k in split:
        print(k, len(split[k]))


if __name__=='__main__':
    files = os.listdir(args.data_dir)
    build_split()

    house_ids = list(set([x.split('_')[0] for x in files]))
    qs_for_houses = {}
    for hid in house_ids:
        get_qs_for_house(hid)

    total_q_count = 0
    unique_q_count = 0
    all_classes = set([])
    # Counts per question template
    count_per_template_unique = { 'total': 0 }
    count_per_template_all = { 'total': 0 }
    ans_per_template_all = { 'total' : 0 }
    templates = list(sorted(QuestionGenerator(0).q_templates_eval_nodes.keys()))

    for q_id in qs_for_houses:
        q_id_count = len(qs_for_houses[q_id].keys())
        q_id_template = str(q_id) + '-' + templates[q_id]

        count_per_template_unique[q_id_template] = q_id_count
        count_per_template_unique['total'] += q_id_count
        count_per_template_all[q_id_template] = 0
        ans_per_template_all[q_id_template] = {}
        unique_q_count += q_id_count

        for q in qs_for_houses[q_id]:
            for ans in qs_for_houses[q_id][q]:
                # Update list of classes
                all_classes.add(str(ans))

                add = qs_for_houses[q_id][q][ans]
                total_q_count += add
                count_per_template_all[q_id_template] += add
                count_per_template_all['total'] += add

                if not ans in ans_per_template_all[q_id_template]:
                    ans_per_template_all[q_id_template][ans] = 0
                ans_per_template_all[q_id_template][ans] += add
                ans_per_template_all['total'] += add

    all_classes_dict = {}
    count = 0
    for cl in sorted(list(all_classes)):
        all_classes_dict[cl] = count
        count += 1
    pp.pprint(all_classes_dict)
    print('Number of classes:', len(all_classes))

    print('>>> STATS:', total_q_count, 'QUESTIONS,', unique_q_count, 'UNIQUE ONES.')
    print("UNIQUE QUESTIONS GENERATED PER TEMPLATE")
    pp.pprint(count_per_template_unique)
    print("ALL QUESTIONS GENERATED PER TEMPLATE")
    pp.pprint(count_per_template_all)
    print("ANSWER DISTRIBUTION PER TEMPLATE")
    pp.pprint(ans_per_template_all)
