import argparse
import os
import random
import time

import numpy as np

from dijkstar import NoPathError
from House3D import objrender, Environment, load_config
from house3d import House3DUtils

from engine import QuestionEngine
from question_gen import QuestionGenerator
from trajectory_gen import TrajectoryGenerator


parser = argparse.ArgumentParser()

parser.add_argument('--house_list', type=str,
                    default='/local/scratch/ccc53/ViQA/scripts/houses.txt')
parser.add_argument('--hid_lo', type=int)
parser.add_argument('--hid_hi', type=int)
parser.add_argument('--examples_per_house', type=int)

parser.add_argument('--suncg_dir', type=str,
                    default='/local/scratch/ccc53/home/SUNCG')
parser.add_argument('--suncgtoolbox_dir', type=str,
                    default='/local/scratch/ccc53/home/SUNCGtoolbox')
parser.add_argument('--house_graph_dir', type=str,
                    default='/local/scratch/ccc53/ViQA/data/graphs')
parser.add_argument('--question_dir', type=str,
                    default='/local/scratch/ccc53/ViQA/data/questions')
parser.add_argument('--video_dir', type=str,
                    default='/local/scratch/ccc53/ViQA/data/videos')
parser.add_argument('--trajectory_dir', type=str,
                    default='/local/scratch/ccc53/ViQA/data/trajectories')

parser.add_argument('--render_config_path', type=str,
                    default='/local/scratch/ccc53/House3D/tests/config.json')
parser.add_argument('--render_width', type=int, default=208)
parser.add_argument('--render_height', type=int, default=160)

parser.add_argument('--print_generated_questions_after_every', type=int, default=20)
parser.add_argument('--stop_generating_after_n_attempts', type=int, default=40)

args = parser.parse_args()


def remove_files(paths):
    for path in paths:
        try:
            os.remove(path)
        except Exception as e:
            print(path, e)


if __name__=='__main__':

    # Get house ids to generate data for
    house_ids = open(args.house_list, 'r').readlines()[args.hid_lo:args.hid_hi]

    # Initialize renderer API and config
    api = objrender.RenderAPI(w=args.render_width, h=args.render_height, device=0)
    cfg = load_config(args.render_config_path)

    for h in house_ids:
        house_id = h[:-1]
        check_path = os.path.join(args.question_dir, house_id + '.json')
        print(check_path)
        if os.path.exists(check_path):
            print('Already have questions for this house --- SKIPPING')
            continue

        # Create house .obj and .mtl files in SUNCG
        scn2scn_path = os.path.join(args.suncgtoolbox_dir, 'gaps/bin/x86_64/scn2scn')
        house_file_dir = os.path.join(args.suncg_dir, 'house', house_id)
        house_file_prefix = os.path.join(house_file_dir, 'house')

        if not os.path.exists(house_file_prefix + '.json'):
            print('Don\'t have SUNCG .json file for this house', house_id, '--- SKIPPING')
            continue
        os.chdir(house_file_dir)
        os.system(scn2scn_path + ' ' + house_file_prefix + '.json ' + house_file_prefix + '.obj')
        resources = [house_file_prefix + '.mtl', house_file_prefix + '.obj']

        # Generate house graph
        print('### Generating trajectories for house', house_id)
        err = False
        try:
            traj_gen = TrajectoryGenerator(house_id=house_id,
                                           load_graph=False,
                                           traj_dir=args.trajectory_dir)
        except Exception as e:
            print(type(e), e, '--- SKIPPING THIS HOUSE!')
            err = True
            remove_files(resources)
        if err:
            continue

        # Generate trajectories (shortest paths) for current house and save them
        room_types = traj_gen.get_all_valid_room_types()
        trajectories = []
        gen_paths_impossible = False
        for i in range(args.examples_per_house):
            path_found = False
            failed_to_generate_path_count = 0
            while not path_found:
                try:
                    [room1, room2] = random.sample(room_types, 2)
                    path = traj_gen.generate_random_path(room1, room2)
                    env_coors = traj_gen.get_graph_to_env_coors(path)
                    trajectories.append(env_coors)
                    path_found = True
                except KeyboardInterrupt:
                    exit(-1)
                except Exception as e:
                    print(type(e), e)
                    if type(e) == NoPathError:
                        failed_to_generate_path_count += 1
                if failed_to_generate_path_count >= args.stop_generating_after_n_attempts:
                    break

            if not path_found:
                gen_paths_impossible = True
                break

        if gen_paths_impossible:
            remove_files(resources)
            continue

        if len(trajectories) > 0:
            trajectories = np.array(trajectories)
            np.save(os.path.join(args.trajectory_dir, house_id + '.npy'), trajectories)
        else:
            remove_files(resources)
            continue

        # Generate questions for the trajectories
        q_generator = QuestionGenerator(traj_gen=traj_gen)
        q_engine = QuestionEngine(question_generator=q_generator, save_dir=args.question_dir)

        start_time = time.time()
        for i in range(args.examples_per_house):
            print('### Generating question for trajectory %d/%d' % (i, args.examples_per_house))
            q_engine.generate_with_timeout(traj_id=i)

            time_taken = time.time() - start_time
            print('...Total time so far: %dm %ds' % (time_taken // 60, time_taken % 60))
            if (i + 1) % args.print_generated_questions_after_every == 0:
                q_engine.pp.pprint(q_engine.question_set)

        q_engine.dump_dataset()
        # Cleanup data for each house after finished
        remove_files(resources)
