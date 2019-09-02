import argparse
import json
import os
import random

import numpy as np

from trajectory_gen import TrajectoryGenerator

parser = argparse.ArgumentParser()

parser.add_argument('--num_trajectories_per_house', type=int, default=300)
parser.add_argument('--resume_from_house_id', type=str)
parser.add_argument('--traj_save_dir', type=str)

args = parser.parse_args()


# Only process houses which have color info
obj_keys = json.load(open('../obj_colors.json')).keys()
house_ids = list(sorted(list(set([x.split('.')[0] for x in obj_keys]))))

for house_id in house_ids:
    # Resume from a certain house id
    if args.resume_from_house_id >= house_id:
        continue

    # Already generated
    traj_file_path = args.traj_save_dir + house_id + '_trajs.npy'
    if os.path.exists(traj_file_path):
        continue

    print('Generating trajectories for house with id', house_id)
    traj_gen = TrajectoryGenerator(house_id=house_id, traj_dir=None)
    room_types = traj_gen.get_all_valid_room_types()

    trajectories = []
    for i in range(args.num_trajectories_per_house):
        path_found = False
        while not path_found:
            try:
                [room1, room2] = random.sample(room_types, 2)
                print(room1, room2)
                path = traj_gen.generate_random_path(room1, room2)
                env_coors = traj_gen.get_graph_to_env_coors(path)
                trajectories.append(env_coors)
                path_found = True
            except KeyboardInterrupt:
                exit(-1)
            except:
                print('Path not found, trying again...')

    trajectories = np.array(trajectories)
    np.save(traj_file_path, trajectories)
