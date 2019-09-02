import argparse
import json
import os
import time

import cv2
import csv
from numpy import isclose
import numpy as np
from scipy.ndimage.measurements import label

from House3D import objrender, Environment, load_config
from House3D.core import local_create_house
from House3D.objrender import RenderMode
from house3d import House3DUtils
from house_parse import HouseParse

import constants

parser = argparse.ArgumentParser()

# Paths
parser.add_argument('--config_path', type=str, '/path/to/House3D/tests/config.json')
parser.add_argument('--csv_path', type=str, '../obj_colors.json')
parser.add_argument('--obj_color_path', type=str, '../colormap_coarse.csv')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--graph_dir', type=str)
parser.add_argument('--trajectory_dir', type=str)

# Defaults
parser.add_argument('--fps', type=int, default=10)
parser.add_argument('--house_id', type=str, default=None)
parser.add_argument('--h_threshold', type=float, default=3e-1)
parser.add_argument('--v_threshold', type=float, default=2e-1)
parser.add_argument('--render_height', type=float, default=160)
parser.add_argument('--render_width', type=float, default=208)

args = parser.parse_args()


# Helper class to call getNearbyPairs() from EmbodiedQA/data/question-gen/engine.py
class ItemInfo():
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta
        self.type = 'object'


class TrajectoryGenerator():

    def __init__(self, traj_dir, house_id=args.house_id, traj_id=None, load_graph=True):

        self.house_id = house_id
        self.traj_dir = traj_dir
        if house_id != None:
            self.create_env(args.config_path, house_id, args.render_width, args.render_height)

        # Contains methods for calculataing distances, room location, etc
        self.hp = HouseParse(dataDir=args.data_dir)
        # Loads house graph and generates shortest paths
        self.utils = House3DUtils(
            self.env,
            rotation_sensitivity=45,
            target_obj_conn_map_dir=False,
            # Changed load_graph method to use pickle directly and Graph(g) initialisation;
            # self.graph.load(path) left the graph empty!
            build_graph=load_graph,
            graph_dir=args.graph_dir)

        self.house = {}
        self.current_room = None

        self.env_coors = None
        self.traj_id = None
        if traj_id != None:
            self.update_trajectory(traj_id)


    """
    Initialize environment for a given house.
    """
    def create_env(self,
                   config_path,
                   house_id,
                   render_width=args.render_width,
                   render_height=args.render_height):
        api = objrender.RenderAPIThread(w=render_width, h=render_height, device=0)
        cfg = load_config(config_path)
        self.env = Environment(api, house_id, cfg)


    """
    Load given trajectory from file.
    """
    def update_trajectory(self, traj_id):
        self.traj_id = traj_id
        load_path = os.path.join(self.traj_dir, self.house_id + '.npy')
        self.env_coors = np.load(load_path)[traj_id]

        # Add look-arounds when entering rooms
        print('Preprocessing trajectory for room views (90 degrees left and right)')
        self.house = {}
        self.add_180s_to_trajectory()


    """
    Update the agent's position.
    """
    def update_env(self, new_pos):
        self.env.cam.pos.x = new_pos[0]
        self.env.cam.pos.y = 1.2
        self.env.cam.pos.z = new_pos[2]
        self.env.cam.yaw = new_pos[3]

        self.env.cam.updateDirection()


    """
    Preprocesses trajectory by adding frames where the agent looks around when entering a room.
    """
    def add_180s_to_trajectory(self):
        # Index house rooms first
        self.build_rooms_and_objects_description()

        new_coors = []
        for i in range(len(self.env_coors)):
            self.update_env(self.env_coors[i])
            new_coors.append(self.env_coors[i])

            # Entered a new room, look around
            if self.update_current_room(self.env_coors[i]):
                init_yaw = new_coors[-1][3]
                new_coor = new_coors[-1]

                # Look around (left, right) up to 90 degrees in increments of 30
                yaw_adds = [1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1]
                for yaw_add in yaw_adds:
                    new_coor = (new_coor[0], new_coor[1], new_coor[2], new_coor[3] + 30 * yaw_add)
                    new_coors.append(new_coor)

        self.env_coors = new_coors


    """
    Render the trajectory step by step.
    """
    def render_and_save_trajectory(self, frame_dir):
        self.env.set_render_mode(RenderMode.RGB)

        s = self.env_coors[0]
        d = self.env_coors[-1]
        assert os.path.exists(frame_dir), 'Can\'t save frames, non-existent directory!'

        filename = '%s/%s_%04d.mp4' % (frame_dir, self.house_id, self.traj_id)
        print('Generating', filename)
        video = cv2.VideoWriter(filename,
                                cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (args.render_width,
                                                                            args.render_height))

        for i in range(len(self.env_coors)):
            self.update_env(self.env_coors[i])
            img = np.array(self.env.render(), copy=False)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Write video frame
            video.write(img)

        cv2.destroyAllWindows()
        video.release()


    """
    Render the agent's current view in a given mode.
    """
    def render_frame(self, mode=RenderMode.RGB):
        self.env.set_render_mode(mode)
        img = self.env.render()

        if mode in [RenderMode.SEMANTIC, RenderMode.RGB]:
            img = np.array(img, copy=False, dtype=np.int32)
            return img
        elif mode == RenderMode.DEPTH:
            img = img[:,:,0]
            img = np.array(img, copy=False, dtype=np.float32)
            return img
        else:
            return None


    """
    Returns a set of all the simple room types in the given list, with the following changes:
        - guest_room -> bedroom
        - toilet -> bathroom
    """
    @staticmethod
    def get_room_types(types):
        room_types = types

        if 'toilet' in room_types:
            room_types.remove('toilet')
            if not 'bathroom' in room_types:
                room_types.append('bathroom')

        if 'guest_room' in room_types:
            room_types.remove('guest_room')
            if not 'bedroom' in room_types:
                room_types.append('bedroom')

        return list(sorted(room_types))


    """
    Trajectory/question generation requires frame-by-frame semantic processing and establishing
    object/room attributes. As we have access to the ground truth information, we can just lookup
    the properties in the dict that this method returns.
    """
    def build_rooms_and_objects_description(self):
        obj_id_to_color = json.load(open(args.obj_color_path))
        room_unique_id = 0

        for room in self.utils.rooms:
            # Add room type to dict
            room_types = TrajectoryGenerator.get_room_types(room['type'])
            room_type = '|'.join(room_types)
            if not room_type in self.house:
                self.house[room_type] = {
                    'room_list': [],
                    'count': 0,
                    'been_here_count': 0
                }
            self.house[room_type]['count'] += 1

            # Prepare property container for room
            room_unique_id += 1
            room_desc = {
                'been_here': False,
                'room_type': room_type,
                'bbox': room['bbox'],
                'objects': {},
                'room_id': room_type + str(room_unique_id)
            }

            objects_in_room = self.get_object_objects_in_room(room)
            for obj in objects_in_room:
                if not obj['coarse_class'] in constants.query_objects:
                    continue

                # Add object type to dict
                obj_type = obj['coarse_class']
                if not obj_type in room_desc['objects']:
                    room_desc['objects'][obj_type] = {
                        'obj_list': [],
                        'count': 0,
                        'seen_count': 0
                    }
                room_desc['objects'][obj_type]['count'] += 1

                # Prepare property container for object
                color = None
                node = '.0_' + obj['id'][2:]
                if self.house_id + node in obj_id_to_color:
                    color = obj_id_to_color[self.house_id + node]
                obj_desc = {
                    'node': node,
                    'bbox': obj['bbox'],
                    'color': color,
                    'seen': False,
                    'room_location': room_type,
                    'obj_type': obj_type,
                    'room_id': room_type + str(room_unique_id)
                }
                room_desc['objects'][obj_type]['obj_list'].append(obj_desc)

            self.house[room_type]['room_list'].append(room_desc)


    """
    Index all objects in the given room.
    """
    def get_object_objects_in_room(self, room):
        return [self.utils.objects['0_' + str(node)] for node in room['nodes']
                if '0_' + str(node) in self.utils.objects and
                self.utils.objects['0_' + str(node)]['coarse_class'] in constants.query_objects]


    """
    Generate a shortest path between random locations in room1 and room2.
    """
    def generate_random_path(self, room1, room2):
        # Disabled assert in getRandomLocation for ALLOWED_TARGET_ROOM_TYPES
        c1 = self.env.house.getRandomLocation(room1, return_grid_loc=True, mixedTp=True) + (0,)
        c2 = self.env.house.getRandomLocation(room2, return_grid_loc=True, mixedTp=True) + (0,)

        print('Generating random path from %s to %s' % (room1, room2), c1, c2)
        start = time.time()
        path = self.utils.compute_shortest_path(c1, c2)
        print(time.time()-start, 'seconds to generate path of length', len(path.nodes))

        return path


    """
    Turn grid path into a trajectory inside the house.
    """
    def get_graph_to_env_coors(self, path):
        env_coors = []
        for node in path.nodes:
            to_coor = self.env.house.to_coor(node[0], node[1])
            env_coor = (to_coor[0], self.env.house.robotHei, to_coor[1], node[2])
            env_coors.append(env_coor)

        return env_coors


    """
    Returns whether the agent entered a new room.
    """
    def update_current_room(self, agent_pos):
        agent_new_pos_obj = TrajectoryGenerator.get_agent_pos_obj(agent_pos)
        if self.current_room == None or \
           not (self.hp.isContained(self.current_room, agent_new_pos_obj, axis=0) and \
                self.hp.isContained(self.current_room, agent_new_pos_obj, axis=2)):
            # New room
            for room_type in self.house:
                for room in self.house[room_type]['room_list']:
                    if self.hp.isContained(room, agent_new_pos_obj, axis=0) and \
                       self.hp.isContained(room, agent_new_pos_obj, axis=2):
                        self.current_room = room
                        # print("New room entered:", self.current_room['room_id'])

                        self.current_room['been_here'] = True
                        self.house[self.current_room['room_type']]['been_here_count'] += 1
                        return True

        return False


    """
    Given a door node, finds at most two rooms which are on either side of the door and adds this
    information to the door object.
    """
    def find_adjacent_rooms_for_door(self, door_node):
        door_obj = self.doors[door_node]

        # One adjacent room is the room which the door belongs to
        door_obj['adjacent_rooms'] = [door_obj['room_id']]
        # If the door doesn't belong to the room we're currently in, append the latter to the list
        # as the other adjacent room
        if door_obj['room_id'] != self.current_room['room_id']:
            door_obj['adjacent_rooms'].append(self.current_room['room_id'])
            return

        # Looking for the second adjacent room
        for room_type in self.house:
            for room_obj in self.house[room_type]['room_list']:
                if room_obj['room_id'] == door_obj['room_id']:
                    continue
                if self.hp.isContained(room_obj, door_obj, axis=0) or\
                   self.hp.isContained(room_obj, door_obj, axis=2):
                    # Found the other adjacent room
                    door_obj['adjacent_rooms'].append(room_obj['room_id'])
                    return


    """
    Given a list of objects in the current view (type->count, approx_depths) and the ground truth
    information, mark doors that were seen. Unlike match_seen_to_ground_truth(), this method tries
    to find seen doors in the entire house. (Uses approximation when computing distances, not 100%
    accurate.)
    """
    def match_seen_to_doors(self, objs_in_frame, agent_pos):
        if 'door' not in objs_in_frame:
            return []

        agent_pos_obj = TrajectoryGenerator.get_agent_pos_obj(agent_pos)
        doors_seen = []
        count = objs_in_frame['door']['count']
        depths = objs_in_frame['door']['depths']

        for i in range(count):
            curr_depth = depths[i]

            for room_type in self.house:
                for room in self.house[room_type]['room_list']:
                    if not 'door' in room['objects']:
                        continue

                    door_objs_in_room = room['objects']['door']['obj_list']
                    for door_obj in door_objs_in_room:
                        # Distance from agent to bbox centre of door
                        coord_bbox = list((np.array(door_obj['bbox']['min']) +\
                                           np.array(door_obj['bbox']['max'])) / 2)
                        bbox_centre = { 'bbox': {'min': coord_bbox, 'max': coord_bbox} }
                        true_dist1 = self.hp.getClosestDistance(agent_pos_obj, bbox_centre)
                        # Default distance computation
                        true_dist2 = self.hp.getClosestDistance(agent_pos_obj, door_obj)

                        # Check if the door object corresponds to the door seen
                        if isclose(curr_depth, true_dist1, rtol=0.25) or\
                           isclose(curr_depth, true_dist2, rtol=0.25):
                           doors_seen.append(door_obj)
                           break

        return doors_seen


    """
    Given a list of objects in the current view (type->count, approx_depths) and the ground truth
    information, mark objects that were seen. (Uses approximation when computing distances, not 100%
    accurate.)
    """
    def match_seen_to_ground_truth(self, objs_in_frame, agent_pos):
        if self.current_room == None:
            return []

        agent_pos_obj = TrajectoryGenerator.get_agent_pos_obj(agent_pos)

        obj_nodes_seen = []
        for obj_type in objs_in_frame:
            count = objs_in_frame[obj_type]['count']
            depths = objs_in_frame[obj_type]['depths']

            for i in range(count):
                if not obj_type in self.current_room['objects']:
                    continue

                curr_depth = depths[i]
                for j in range(self.current_room['objects'][obj_type]['count']):
                    # An object is usually visible across multiple frames - only mark it once
                    if not self.current_room['objects'][obj_type]['obj_list'][j]['seen']:
                        # Get distance between bboxes of agent and object
                        coord1 = list((np.array(
                            self.current_room['objects'][obj_type]['obj_list'][j]['bbox']['min']) +\
                                      np.array(
                            self.current_room['objects'][obj_type]['obj_list'][j]['bbox']['max']))
                                     / 2)
                        bbox_centre = { 'bbox': { 'min': coord1, 'max': coord1 } }
                        true_dist1 = self.hp.getClosestDistance(agent_pos_obj, bbox_centre)

                        true_dist2 = self.hp.getClosestDistance(
                            agent_pos_obj,
                            self.current_room['objects'][obj_type]['obj_list'][j])

                        # Check if the object in view is in the room
                        if isclose(curr_depth, true_dist1, rtol=0.25) or\
                           isclose(curr_depth, true_dist2, rtol=0.25):
                            obj_nodes_seen.append(
                                self.current_room['objects'][obj_type]['obj_list'][j])
                            self.current_room['objects'][obj_type]['obj_list'][j]['seen'] = True
                            self.current_room['objects'][obj_type]['seen_count'] += 1
                            break

        return obj_nodes_seen


    """
    Generate a trajectory and gather seen rooms and objects. Optionally returns objects in all video
    frames corresponding to the trajectory.
    """
    def generate_trajectory_and_seen_items(self,
                                           frame_dir=None,
                                           compute_seen_doors=False,
                                           return_objects_in_frames=False):
        self.build_rooms_and_objects_description()

        if frame_dir:
            self.render_and_save_trajectory(frame_dir)

        rgb_to_obj = TrajectoryGenerator.get_semantic_to_object_mapping(args.csv_path)
        self.current_room = None

        if return_objects_in_frames:
            objects_in_frames = []

        # Parse frames
        start = time.time()
        for c in range(len(self.env_coors)):
            # Update agent position in the environment
            self.update_env(self.env_coors[c])
            semantic_img = self.render_frame(mode=RenderMode.SEMANTIC)
            depth_img = self.render_frame(mode=RenderMode.DEPTH)

            # Mark current room as visited
            self.update_current_room(self.env_coors[c])

            # Get objects types and approximate depths from current frame
            objs_in_frame = TrajectoryGenerator.get_objects_in_frame(semantic_img,
                                                                     rgb_to_obj,
                                                                     depth_img)
            if return_objects_in_frames:
                objects_in_frames.append(list(objs_in_frame.keys()))

            # Mark objects in ground truth room that correspond to current view
            seen_nodes = self.match_seen_to_ground_truth(objs_in_frame, self.env_coors[c])

            if compute_seen_doors:
                # Store objects corresponding to all seen doors (in the entire house)
                seen_doors = self.match_seen_to_doors(objs_in_frame, self.env_coors[c])
                for door in seen_doors:
                    if door['node'] in self.doors:
                        continue
                    # See which rooms are on both sides of the new door
                    self.doors[door['node']] = door
                    self.find_adjacent_rooms_for_door(door['node'])

        print(time.time()-start, 'seconds to process', str(len(self.env_coors)), 'frames.')

        if return_objects_in_frames:
            return objects_in_frames


    """
    Returns a list of the valid room types in the house.
    """
    def get_all_valid_room_types(self):
        return [x for x in self.env.house.all_roomTypes if TrajectoryGenerator.valid_room_type(x)]


    """
    Get pairs of nearby objects for a visited room with marked objects.
    """
    def get_nearby_object_pairs(self, room_desc):
        assert room_desc['been_here'], 'This room has not been visited!'

        obj_container = []
        for obj_type in room_desc['objects']:
            cnt_type = 0
            for obj_entry in room_desc['objects'][obj_type]['obj_list']:
                # Make sure the object was seen on the trajectory
                if not obj_entry['seen']:
                    continue
                cnt_type += 1
                obj = ItemInfo(name=obj_type + str(cnt_type), meta=obj_entry)
                obj_container.append(obj)

        if len(obj_container) > 0:
            return self.hp.getNearbyPairs(obj_container,
                                          hthreshold=args.h_threshold,
                                          vthreshold=args.v_threshold)
        return {'on': [], 'next_to': []}


    """
    Returns a dict with keys ['on', 'next_to'] and values as list of tuples (obj1, obj2, dist)
    showing spatial relationships between objects.
    """
    def get_all_nearby_object_pairs(self):
        all_pairs = {'on': [], 'next_to': []}

        for room_type in self.house:
            for room_obj in self.house[room_type]['room_list']:
                # Only look at visited rooms
                if room_obj['been_here']:
                    pairs = self.get_nearby_object_pairs(room_obj)
                    for rel in ['on', 'next_to']:
                        all_pairs[rel] += pairs[rel]

        return all_pairs


    """
    Get the list of all objects (either on the trajectory or in the entire house).
    """
    def get_all_objects(self, include_unseen_objects=False, include_objects_in_all_rooms=False):
        obj_list = []

        for room_type in self.house:
            for room in self.house[room_type]['room_list']:
                if not room['been_here'] and not include_objects_in_all_rooms:
                    continue
                for obj_type in room['objects']:
                    for obj in room['objects'][obj_type]['obj_list']:
                        if obj['seen'] or include_unseen_objects:
                            obj_list.append(obj)

        return obj_list


    """
    Get the list of all rooms (either on the trajectory or in the entire house). Does not include
    object list for rooms.
    """
    def get_all_rooms(self, include_unseen_rooms=False):
        room_list = []

        for room_type in self.house:
            for room in self.house[room_type]['room_list']:
                if room['been_here'] or include_unseen_rooms:
                    # Don't include object list
                    room_list.append({
                        'been_here': True,
                        'room_type': room_type,
                        'bbox': room['bbox'],
                        'room_id': room['room_id']
                    })

        return room_list


    """
    Return agent's current position inside an object with 'bbox' attribute. Useful for calling
    HouseParse methods.
    """
    @staticmethod
    def get_agent_pos_obj(agent_pos):
        agent_new_pos_obj = {
            'bbox': {
                'min': agent_pos[0:3],
                'max': agent_pos[0:3],
            },
        }
        return agent_new_pos_obj


    """
    Given a depth map and an image with numbered disjoint regions corresponding to a single object
    type, return an __approximate__ depth for each one of them.
    """
    @staticmethod
    def get_approx_depths_for_object_type(depth_img, labeled_objs_img, num_objs):
        approx_depths = []

        for i in range(num_objs):
            first_idx = next(idx for idx, val in np.ndenumerate(labeled_objs_img) if val==i+1)
            approx_depths.append(depth_img[first_idx] / 255.0 * 20.0)

        return approx_depths


    """
    Extract objects from a frame.
    object_type -> (num_objects, object_depths)
    """
    @staticmethod
    def get_objects_in_frame(semantic_img, rgb_to_obj, depth_img):
        label_img = TrajectoryGenerator.rgb_to_int_image(semantic_img)
        labels = np.unique(label_img)

        objs_in_frame = {}
        # Process information about each unique type of object in the current frame
        only_curr_obj_img = np.zeros(shape=label_img.shape, dtype=np.int32)
        for i in range(len(labels)):
            # "Paint" on a background only the objects of the current type
            only_curr_obj_img[:,:] = 0
            only_curr_obj_img[np.where(label_img == labels[i])] = 1

            # Find number of occurrences of object in frame (might be misleading, occlusions or
            # several objects overlapping e.g. chairs)
            s = [[1,1,1],
                 [1,1,1],
                 [1,1,1]]
            num_objs = label(only_curr_obj_img, output=only_curr_obj_img)

            # Find semantic color of object and convert it to object type
            first_idx = next(idx for idx, val in np.ndenumerate(only_curr_obj_img) if val==1)
            rgb = (semantic_img[first_idx[0], first_idx[1], 0],
                   semantic_img[first_idx[0], first_idx[1], 1],
                   semantic_img[first_idx[0], first_idx[1], 2])
            obj_name = rgb_to_obj[rgb]

            # Check if we want to ask questions about the object type
            if not obj_name in constants.query_objects:
                continue

            objs_in_frame[obj_name] = {}
            # Get number of objects of this type in the frame
            objs_in_frame[obj_name]['count'] = num_objs

            # Get approximate depths
            depths = TrajectoryGenerator.get_approx_depths_for_object_type(depth_img,
                                                                           only_curr_obj_img,
                                                                           num_objs)
            objs_in_frame[obj_name]['depths'] = depths

        return objs_in_frame


    """
    Open CSV category file to map semantic frames to sets of objects.
    """
    @staticmethod
    def get_semantic_to_object_mapping(path):
        f = open(path, newline="")
        reader = csv.DictReader(f)

        all_objects_dict = {}
        for line in reader:
            all_objects_dict[(int(line['r']), int(line['g']), int(line['b']))] = line['name']

        return all_objects_dict


    """
    Map RGB values in an image to integers: (r,g,b) -> (256^2 * r + 256 * g + b).
    """
    @staticmethod
    def rgb_to_int_image(img):
        out = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.int32)
        out = (img[:,:,0] << 16) | (img[:,:,1] << 8) | (img[:,:,2])
        return out


    """
    Blacklist some room types.
    """
    @staticmethod
    def valid_room_type(room_type):
        return len(room_type) > 0 and\
               all([not tp.lower() in constants.exclude_rooms for tp in room_type])
