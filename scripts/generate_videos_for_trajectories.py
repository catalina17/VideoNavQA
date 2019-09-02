import argparse
import json
import os

from trajectory_gen import TrajectoryGenerator

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str)
parser.add_argument('--suncg_dir', type=str,
                    default='/local/scratch/ccc53/home/SUNCG')
parser.add_argument('--suncgtoolbox_dir', type=str,
                    default='/local/scratch/ccc53/home/SUNCGtoolbox')
parser.add_argument('--trajectory_dir', type=str)
parser.add_argument('--video_dir', type=str)
parser.add_argument('--video_lengths_dir', type=str)
parser.add_argument('--start_house_idx', type=int)
parser.add_argument('--end_house_idx', type=int)

args = parser.parse_args()


datafiles = list(sorted(os.listdir(args.data_dir)))
print('Resuming from house', args.start_house_idx)

cwd = os.getcwd()
curr_idx = args.start_house_idx
for datafile in datafiles[curr_idx:]:
    house_id = datafile.split('.')[0]
    if curr_idx == args.end_house_idx:
        print('Reached house', args.end_house_idx, 'with id', house_id, '--- stopping.')
        break
    curr_idx += 1

    print('House', house_id)
    with open(os.path.join(args.data_dir, datafile), 'r') as f:
        data = json.load(f)

    # Create house .obj and .mtl files in SUNCG
    scn2scn_path = os.path.join(args.suncgtoolbox_dir, 'gaps/bin/x86_64/scn2scn')
    house_file_dir = os.path.join(args.suncg_dir, 'house', house_id)
    house_file_prefix = os.path.join(house_file_dir, 'house')

    if not os.path.exists(house_file_prefix + '.json'):
        print('Don\'t have SUNCG .json file for house', house_id, '--- SKIPPING')
        continue
    os.chdir(house_file_dir)
    os.system(scn2scn_path + ' ' + house_file_prefix + '.json ' + house_file_prefix + '.obj')
    os.chdir(cwd)

    v_lens = {}
    traj_gen = TrajectoryGenerator(house_id=house_id,
                                   traj_id=0,
                                   load_graph=False,
                                   traj_dir=args.trajectory_dir)
    for traj_id in sorted(data.keys()):
        traj_gen.update_trajectory(int(traj_id))

        example_id = '%s_%04d' % (house_id, int(traj_id))
        v_lens[example_id] = len(traj_gen.env_coors)
        traj_gen.render_and_save_trajectory(frame_dir=args.video_dir)

    os.remove(house_file_prefix + '.obj')
    with open(os.path.join(args.video_lengths_dir, house_id + '.json'), 'w') as out:
        json.dump(v_lens, out)
