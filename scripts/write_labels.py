import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str)
parser.add_argument('--labels_file', type=str)
parser.add_argument('--q_ids_file', type=str)

args = parser.parse_args()


label_to_class_v3 = {
    '1': 0,
    '10': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'False': 10,
    'True': 11,
    'bathroom': 12,
    'bathroom|bedroom': 13,
    'bathtub': 14,
    'bed': 15,
    'bedroom': 16,
    'black': 17,
    'blue': 18,
    'brown': 19,
    'chair': 20,
    'child_room': 21,
    'clock': 22,
    'computer': 23,
    'curtain': 24,
    'desk': 25,
    'dining_room': 26,
    'dining_room|kitchen': 27,
    'dining_room|kitchen|living_room': 28,
    'dining_room|kitchen|living_room|office': 29,
    'dining_room|living_room': 30,
    'dining_room|living_room|office': 31,
    'door': 32,
    'dresser': 33,
    'entryway': 34,
    'green': 35,
    'grey': 36,
    'gym': 37,
    'gym_equipment': 38,
    'gym|living_room': 39,
    'gym|living_room|office': 40,
    'hallway': 41,
    'hanging_kitchen_cabinet': 42,
    'heater': 43,
    'kitchen': 44,
    'kitchen_cabinet': 45,
    'kitchen|living_room': 46,
    'living_room': 47,
    'living_room|office': 48,
    'maroon': 49,
    'mirror': 50,
    'office': 51,
    'ottoman': 52,
    'rug': 53,
    'shower': 54,
    'sink': 55,
    'sofa': 56,
    'stand': 57,
    'switch': 58,
    'table': 59,
    'tan': 60,
    'teal': 61,
    'television': 62,
    'toilet': 63,
    'tv_stand': 64,
    'vase': 65,
    'vehicle': 66,
    'wardrobe': 67,
    'wardrobe_cabinet': 68,
    'white': 69
}

datafiles = sorted(os.listdir(args.data_dir))
labels = {}
q_ids = {}

for datafile in datafiles:
    data = None
    with open(os.path.join(args.data_dir, datafile), 'r') as f:
        data = json.load(f)

    for traj_id in data:
        example_id = '%s_%04d' % (datafile.split('.')[0], int(traj_id))

        label = data[traj_id]['q_ans']
        q_id = data[traj_id]['q_id']

        q_ids[example_id] = q_id
        assert str(label) in label_to_class_v3, "Label " + str(label) + " not found!"
        labels[example_id] = label_to_class_v3[str(label)]

with open(args.labels_file, 'w') as f:
   json.dump(labels, f)
with open(args.q_ids_file, 'w') as f:
    json.dump(q_ids, f)
