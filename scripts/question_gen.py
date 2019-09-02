from enum import Enum
import operator
import os
import random
import re

from constants import *
from question_build import QuestionBuild
from trajectory_gen import TrajectoryGenerator


class QuestionGenerationError(Exception):
    def __init__(self, msg=None):
        super(QuestionGenerationError, self).__init__(msg)


class ItemType(Enum):
    ROOMS = 1
    OBJECTS = 2
    OBJECTS_REL = 3


SET_SIZES = [2,3]


class QuestionGenerator():

    tags_to_instantiate = ['attr', 'obj_type', 'room_type', 'color', 'rel', 'comp', 'comp_rel',
                           'comp_sup']

    q_templates_eval_nodes = {
        'Are the <attr1> <obj_type1> and the <attr2> <obj_type2> the same color?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type1>', 'filter.<attr1>', 'unique', 'get_attr.color'],
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_attr.color'],
                ['tree_0|tree_1', 'equal']
            ],
            'ans_type': bool
        },

        'Are both the <attr1> <obj_type1> and the <attr2> <obj_type2> <color>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type1>', 'filter.<attr1>', 'unique', 'get_attr.color'],
                ['tree_0|<color>', 'equal'],
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_attr.color'],
                ['tree_2|<color>', 'equal'],
                ['tree_1|tree_3', 'logical_and']
            ],
            'ans_type': bool
        },

        'Are all <attr> <obj_type-pl> <color>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'continue_if_non_empty', 'get_attr.color'],
                ['tree_0|<color>', 'equal_set']
            ],
            'ans_type': bool
        },

        'Is the <attr1> thing <rel> the <attr2> <obj_type2> <art> <obj_type1>?': {
            'inputs': [ItemType.OBJECTS_REL],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_rel_objects.<rel>',
                 'filter.<attr1>', 'unique', 'get_attr.obj_type'],
                ['tree_0|<obj_type1>', 'equal']
            ],
            'ans_type': bool
        },

        'Are all <attr> things <obj_type-pl>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.<attr>', 'continue_if_non_empty', 'get_attr.obj_type'],
                ['tree_0|<obj_type>', 'equal_set']
            ],
            'ans_type': bool
        },

        'Are both the <attr1> <obj_type1> and the <attr2> <obj_type2> in the <room_type>?': {
            'inputs': [ItemType.ROOMS, ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter_unwanted_rooms', 'filter.room_type.<room_type>', 'unique', 'get_attr.room_type'],
                ['inputs_1', 'filter.obj_type.<obj_type1>', 'filter.<attr1>', 'unique', 'get_attr.room_location'],
                ['tree_0|tree_1', 'equal'],
                ['inputs_1', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_attr.room_location'],
                ['tree_0|tree_3', 'equal'],
                ['tree_2|tree_4', 'logical_and']
            ],
            'ans_type': bool
        },

        'Are all <attr> <obj_type-pl> in the <room_type>?': {
            'inputs': [ItemType.ROOMS, ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter_unwanted_rooms', 'filter.room_type.<room_type>', 'unique', 'get_attr.room_type'],
                ['inputs_1', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'continue_if_non_empty', 'get_attr.room_location'],
                ['tree_1|tree_0', 'equal_set']
            ],
            'ans_type': bool
        },

        'Is the <attr1> <obj_type> <comp_rel> than the <attr2> one?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr1>', 'unique'],
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr2>', 'unique'],
                ['tree_0|tree_1', 'continue_if_distinct'],
                ['tree_0|tree_1', 'comp_rel']
            ],
            'ans_type': bool
        },

        'Is the <room_type1> <comp_rel> than the <room_type2>?': {
            'inputs': [ItemType.ROOMS],
            'tree': [
                ['inputs_0', 'filter_unwanted_rooms', 'filter.room_type.<room_type1>', 'unique'],
                ['inputs_0', 'filter_unwanted_rooms', 'filter.room_type.<room_type2>', 'unique'],
                ['tree_0|tree_1', 'comp_rel']
            ],
            'ans_type': bool
        },

        'Are there as many <attr1> <obj_type1-pl> as there are <attr2> <obj_type2-pl>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type1>', 'filter.<attr1>', 'count_exists'],
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'count_exists'],
                ['tree_0|tree_1', 'equal']
            ],
            'ans_type': bool
        },

        'Are there <comp> <attr1> <obj_type1-pl> than <attr2> <obj_type2-pl>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type1>', 'filter.<attr1>', 'count_exists'],
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'count_exists'],
                ['tree_0|tree_1', 'comp.<comp>']
            ],
            'ans_type': bool
        },

        'Is there <art> <attr> <obj_type>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'exists']
            ],
            'ans_type': bool
        },

        'Is there set(<art> <attr{}> <obj_type{}>)?': {
            'inputs': [ItemType.OBJECTS],
            'iter_set_fn_list': ['filter.obj_type.<obj_type{}>', 'filter.<attr{}>'],
            'set_size': (lambda: random.choice(SET_SIZES)),
            'tree': [
                ['inputs_0', 'iter_set', 'exists_set']
            ],
            'ans_type': bool
        },

        'Is there set(<art> <attr{}> <obj_type{}>) in the <room_type>?': {
            'inputs': [ItemType.ROOMS, ItemType.OBJECTS],
            'iter_set_fn_list': ['filter.obj_type.<obj_type{}>', 'filter.<attr{}>', 'continue_if_non_empty', 'get_attr.room_location'],
            'set_size': (lambda: random.choice(SET_SIZES)),
            'tree': [
                ['inputs_0', 'filter_unwanted_rooms', 'filter.room_type.<room_type>', 'unique', 'get_attr.room_type'],
                ['inputs_1', 'iter_set', 'flatten_set'],
                ['tree_1|tree_0', 'equal_set']
            ],
            'ans_type': bool
        },

        'Is there <art> <room_type>?': {
            'inputs': [ItemType.ROOMS],
            'tree': [
                ['inputs_0', 'filter_unwanted_rooms', 'filter.room_type.<room_type>', 'exists']
            ],
            'ans_type': bool
        },

        'Is there a room that has set(<art> <attr{}> <obj_type{}>)?': {
            'inputs': [ItemType.OBJECTS],
            'iter_set_fn_list': ['filter.obj_type.<obj_type{}>', 'filter.<attr{}>', 'continue_if_non_empty', 'get_attr.room_id'],
            'set_size': (lambda: random.choice(SET_SIZES)),
            'tree': [
                ['inputs_0', 'iter_set', 'intersect', 'exists']
            ],
            'ans_type': bool
        },

        'Is there set(<art> <room_type{}>)?': {
            'inputs': [ItemType.ROOMS],
            'iter_set_fn_list': ['filter.room_type.<room_type{}>'],
            'set_size': (lambda: random.choice(SET_SIZES)),
            'tree': [
                ['inputs_0', 'filter_unwanted_rooms', 'iter_set', 'exists_set']
            ],
            'ans_type': bool
        },

        'How many <obj_type-pl> are <attr>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'count']
            ],
            'ans_type': int
        },

        'How many <attr> <obj_type-pl> are in the <room_type>?': {
            'inputs': [ItemType.OBJECTS, ItemType.ROOMS],
            'tree': [
                # Verify room_location is unique
                ['inputs_1', 'filter_unwanted_rooms', 'filter.room_type.<room_type>', 'get_attr.room_id', 'unique'],
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'filter.room_location.<room_type>', 'count']
            ],
            'ans_type': int
        },

        'How many <attr1> <obj_type1-pl> are in the room containing the <attr2> <obj_type2>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type1>', 'filter.<attr1>'],
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_attr.room_id'],
                ['tree_0', 'filter.room_id.<tree_1>', 'count']
            ],
            'ans_type': int
        },

        'How many <room_type-pl> are there?': {
            'inputs': [ItemType.ROOMS],
            'tree': [
                ['inputs_0', 'filter_unwanted_rooms', 'filter.room_type.<room_type>', 'count']
            ],
            'ans_type': int
        },

        'How many rooms have <attr> <obj_type-pl>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'get_attr.room_id', 'count_unique']
            ],
            'ans_type': int
        },

        'What color is the <attr> <obj_type>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'unique', 'get_attr.color']
            ],
            'ans_type': 'color'
        },

        'What color is the <attr1> <obj_type1> <rel> the <attr2> <obj_type2>?': {
            'inputs': [ItemType.OBJECTS_REL],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_rel_objects.<rel>',
                 'filter.obj_type.<obj_type1>', 'filter.<attr1>', 'unique', 'get_attr.color'],
            ],
            'ans_type': 'color'
        },

        'What is the <attr> thing?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.<attr>', 'unique', 'get_attr.obj_type']
            ],
            'ans_type': 'obj_type'
        },

        'What is the <attr1> thing <rel> the <attr2> <obj_type2>?': {
            'inputs': [ItemType.OBJECTS_REL],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_rel_objects.<rel>',
                 'filter.<attr1>', 'unique', 'get_attr.obj_type'],
            ],
            'ans_type': 'obj_type'
        },

        'Where is the <attr> <obj_type>?': {
            'inputs': [ItemType.OBJECTS],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type>', 'filter.<attr>', 'unique', 'get_attr.room_location', 'room_if_allowed']
            ],
            'ans_type': 'room_location'
        },

        'Where is the <attr1> <obj_type1> <rel> the <attr2> <obj_type2>?': {
            'inputs': [ItemType.OBJECTS_REL],
            'tree': [
                ['inputs_0', 'filter.obj_type.<obj_type2>', 'filter.<attr2>', 'unique', 'get_rel_objects.<rel>',
                 'filter.obj_type.<obj_type1>', 'filter.<attr1>', 'unique', 'get_attr.room_location', 'room_if_allowed']
            ],
            'ans_type': 'room_location'
        },

        'Where are the set(<attr{}> <obj_type{}>)?': {
            'inputs': [ItemType.OBJECTS],
            'iter_set_fn_list': ['filter.obj_type.<obj_type{}>', 'filter.<attr{}>', 'unique', 'get_attr.room_id'],
            'set_size': (lambda: random.choice(SET_SIZES)),
            'tree': [
                ['inputs_0', 'iter_set', 'unique_set', 'strip_nums', 'room_if_allowed']
            ],
            'ans_type': 'room_location'
        },
    }


    def __init__(self, traj_gen):
        self.traj_gen = traj_gen
        self.set_size = None
        self.fine_grained_room_counts = False


    """
    Returns the volume of a bbox.
    """
    def bbox_volume(self, box):
        sides = [box['max'][i] - box['min'][i] for i in range(3)]
        volume = sides[0] * sides[1] * sides[2]
        return volume


    """
    Will return true if box1 is larger than box2 by threshold %.
    """
    def bbox_larger(self, box1, box2, threshold):
        volume1 = self.bbox_volume(box1)
        volume2 = self.bbox_volume(box2)
        return ((volume1 - volume2) / volume2) >= threshold


    ##
    ## Template helper functions
    ##
    def equal(self, v1, v2):
        return v1 == v2


    def equal_set(self, set, v):
        return len(set) > 0 and all([v == x for x in set])


    def flatten_set(self, *args):
        flattened = []
        for set in args:
            flattened += set
        return flattened


    def continue_if_distinct(self, obj1, obj2):
        if obj1 == obj2:
            raise QuestionGenerationError('Need two distinct objects of the same type!')
        return [obj1, obj2]


    def continue_if_non_empty(self, set):
        if len(set) > 0:
            return set
        raise QuestionGenerationError('Set is empty!')


    def exists(self, set):
        return len(set) > 0


    def exists_set(self, *args):
        assert len(args) > 0, 'No sets given as input!'
        for set in args:
            if not self.exists(set):
                return False
        return True


    def unique(self, set):
        if len(set) == 0:
            raise QuestionGenerationError('Not a singleton set!')
        if not self.equal_set(set, set[0]):
            raise QuestionGenerationError('Not a singleton set!')
        return set[0]


    def unique_set(self, *args):
        return self.unique(args)


    def intersect(self, *args):
        assert len(args) > 1, 'Not enough sets to intersect!'
        result_set = args[0]
        for i in range(1, len(args)):
            result_set = list(set(result_set).intersection(set(args[i])))
        return result_set


    def count(self, set):
        return len(set)


    def count_exists(self, set):
        if len(set) == 0:
            raise QuestionGenerationError('No items in the set!')
        return len(set)


    def count_unique(self, sett):
        return len(list(set(sett)))


    def comp(self, op, v1, v2):
        assert op in ['more', 'fewer'], 'Invalid operator!'
        if op == 'more':
            op = operator.gt
        elif op == 'fewer':
            op = operator.lt
        return op(v1, v2)


    def comp_rel(self, b1, b2):
        assert 'bbox' in b1 and 'bbox' in b2, 'One of the arguments does not have a bbox field!'
        return self.bbox_larger(b1['bbox'], b2['bbox'], 0.25)


    def comp_sup(self, items):
        if len(items) < 2:
            raise QuestionGenerationError('Need at least 2 items for getting the biggest one!')
        for item in items:
            assert 'bbox' in item, 'At least one item in the set does not have a bbox field!'
        biggest_item = items[0]
        for i in range(1, len(items)):
            if self.bbox_larger(items[i]['bbox'], biggest_item['bbox'], 0.1):
                biggest_item = items[i]
        return biggest_item


    def get_attr(self, attr, item):
        if type(item) != list:
            if not attr in item or item[attr] == None:
                raise QuestionGenerationError(attr + ' attribute not present or None!')
            return item[attr]
        else:
            for x in item:
                if not attr in x or x[attr] == None:
                    raise QuestionGenerationError(attr + ' attribute not present or None!')
            return [x[attr] for x in item]


    def filter(self, attr_name, attr_value, set):
        if self.fine_grained_room_counts and attr_name == 'room_type':
            simple_types = attr_value.split('|')
            res = []
            for x in set:
                if all([simple_type in x['room_type'] for simple_type in simple_types]):
                    res.append(x)
            return res

        if attr_name == None or attr_name == []:
            return set

        if type(attr_name) == list:
            res = set
            for i in range(len(attr_name)):
                res = [x for x in res if x[attr_name[i]] == attr_value[i]]
            return res

        return [x for x in set if x[attr_name] == attr_value]


    def filter_unwanted_rooms(self, set):
        clean_rooms = []
        for x in set:
            assert 'room_type' in x, 'room_type attribute not present!'
            if x['room_type'] in exclude_rooms:
                continue
            clean_rooms.append(x)
        return clean_rooms


    def get_rel_objects(self, rel, obj):
        all_pairs_rel = self.traj_gen.get_all_nearby_object_pairs()[rel]
        if len(all_pairs_rel) == 0:
            raise QuestionGenerationError('No nearby object pairs for ' + rel + '!')

        rel_objects = []
        for tpl in all_pairs_rel:
            if tpl[1].meta == obj:
                rel_objects.append(tpl[0].meta)
        if len(rel_objects) == 0:
            raise QuestionGenerationError('No nearby object pairs for object ' + obj['node'] + '!')
        return rel_objects


    def logical_and(self, v1, v2):
        return v1 and v2


    def strip_nums(self, string):
        string = re.sub('[0-9]', '', string)
        return string


    def strip_pl(self, string):
        if not string.endswith('-pl'):
            return string
        return string[:-3]


    def strip_idx_placeholder(self, string):
        if not "{}" in string:
            return string
        string = re.sub('{}', '', string)
        return string


    def room_if_allowed(self, string):
        if string in exclude_rooms:
            raise QuestionGenerationError('Can\' have this value for room_location/_type!')
        return string


    ##
    ## Question generation
    ##
    """
    Check whether we need to use granular room counts (e.g. 'living room/kitchen/office' counts as a
    'living room/office' and a 'kitchen' and a 'kitchen/office').
    """
    def check_fine_grained_room_counts(self):
        if self.q_template_string in [
            'Are there <comp> <room_type1-pl> than <room_type2-pl>?',
            'How many <room_type-pl> are there?',
            'Is there <art> <room_type>?',
            'Is there set(<art> <room_type{}>)?']:
            self.fine_grained_room_counts = True
        else:
            self.fine_grained_room_counts = False


    """
    Check whether we can instantiate the attribute tag with this type. This is done to avoid leaking
    information about the answer of the question (i.e. we don't want to ask "Where is the brown rug
    in the bedroom?")
    """
    def is_restricted_type(self, attr_type):
        return attr_type.split('_')[0] in self.q_template_string or\
               (attr_type == 'room_type' and 'Where' in self.q_template_string)


    """
    Get a random instantiation of all tags for the current question. Object and room types are
    different. Attribute tags are instantiated from the set of all attributes that correspond to
    object types that have already been chosen.
    """
    def get_next_instantiation(self):
        # Handle object and room tags separately - their values should be distinct
        # Object tag instantiations
        obj_tags = [tag for tag in self.tag_instantiations if 'obj_type' in tag]
        if len(obj_tags) > 0:
            obj_types = list(set([x['obj_type'] for x in self.house['objects']]))

            # Special case for a questions needing object types that are not on this trajectory
            if self.q_template_string == 'Is there set(<art> <attr{}> <obj_type{}>)?' and\
               self.answer == False:
                obj_types += random.sample(query_objects.keys(), len(obj_tags))

            # Don't ask questions about doors unless we can relate other objects to them
            if not '<rel>' in self.q_template_string and 'door' in obj_types:
                obj_types.remove('door')
            if len(obj_tags) > len(obj_types):
                raise QuestionGenerationError('Not enough distinct object types!')

            obj_tag_instantiations = random.sample(obj_types, len(obj_tags))
            for i in range(len(obj_tag_instantiations)):
                self.tag_instantiations[obj_tags[i]]['value'] = obj_tag_instantiations[i]

        # Room tag instantiations
        room_tags = [tag for tag in self.tag_instantiations if 'room_type' in tag]
        if len(room_tags) > 0:
            if self.house['rooms'] != []:
                room_types = list(set([x['room_type'] for x in self.house['rooms']
                                                      if not x['room_type'] in exclude_rooms]))
                # Special case for two questions needing room types that are not on this trajectory
                if self.q_template_string in ['Is there <art> <room_type>?',
                                              'Is there set(<art> <room_type{}>)?'] and\
                   self.answer == False:
                    all_rooms_in_house = self.traj_gen.get_all_rooms(include_unseen_rooms=True)
                    room_types_not_in_house =\
                        [x for x in all_simple_room_types if not x in all_rooms_in_house]
                    room_types += random.sample(room_types_not_in_house, len(room_tags))
            else:
                room_types = list(set(
                    [x['room_location'] for x in self.house['objects']
                     if not x['room_location'] in exclude_rooms]))

            if len(room_tags) > len(room_types):
                raise QuestionGenerationError('Not enough distinct room types!')

            room_tag_instantiations = random.sample(room_types, len(room_tags))
            for i in range(len(room_tag_instantiations)):
                self.tag_instantiations[room_tags[i]]['value'] = room_tag_instantiations[i]

        for tag in self.tag_instantiations:
            if tag == 'rel':
                self.tag_instantiations[tag]['value'] = random.choice(['on', 'next_to'])
            elif tag == 'comp':
                self.tag_instantiations[tag]['value'] = random.choice(['more', 'fewer'])
            elif tag == 'comp_rel':
                self.tag_instantiations[tag]['value'] = 'bigger'
            elif tag == 'comp_sup':
                self.tag_instantiations[tag]['value'] = 'biggest'
            elif 'color' in tag:
                colors = [x['color'] for x in self.house['objects'] if x['color'] != None]
                if len(colors) == 0:
                    raise QuestionGenerationError('No colors available!')
                self.tag_instantiations[tag]['value'] = random.choice(colors)

        for tag in self.tag_instantiations:
            if 'attr' in tag:
                self.tag_instantiations[tag]['value'] = []
                self.tag_instantiations[tag]['type'] = []

                # Attributes describe objects and can only be color or room_location
                for attr_type in ['color', 'room_type']:
                    # Don't leak information about the queried attribute
                    if not self.is_restricted_type(attr_type):
                        # Find corresponding object tag in question template (or just the object
                        # tag)
                        obj_tag = '<obj_type'
                        idx = re.findall('[0-9]', tag)
                        if len(idx) > 0:
                            obj_tag += idx[0]
                        obj_tag += '>'

                        if attr_type == 'room_type':
                            attr_type = 'room_location'
                        if obj_tag in self.tag_instantiations:
                            obj_attrs = [obj[attr_type] for obj in self.house['objects']
                                         if obj[attr_type] != None and\
                                            obj['obj_type'] ==\
                                                self.tag_instantiations[obj_tag]['value']]
                        else:
                            obj_attrs = [obj[attr_type] for obj in self.house['objects']
                                         if obj[attr_type] != None]
                        if len(obj_attrs) == 0:
                            raise QuestionGenerationError('No object attributes to choose from!')
                        attr_value = random.choice(obj_attrs)

                        # Don't always assign all possible attributes, for question variability
                        if random.choice([0, 1]):
                            if attr_type == 'room_location' and attr_value in exclude_rooms:
                                continue
                            self.tag_instantiations[tag]['value'].append(attr_value)
                            self.tag_instantiations[tag]['type'].append(attr_type)

            # Special case where attr1 and attr2 need to be different
            if self.q_template_string ==\
                'Are both the <attr1> thing and the <attr2> thing <obj_type-pl>?':
                if self.tag_instantiations['attr1'] == self.tag_instantiations['attr2']:
                    raise QuestionGenerationError('Can\'t have same attributes for this question!')


    """
    Extracts the function and parameters encoded in the given operation, which is within a branch of
    the template execution tree.
    """
    def split_tokens_and_instantiate(self, branch_op):
        self.fn_inputs = []
        self.tokens_fn_call = branch_op.split('.')

        # A tag never occurs before the last token
        for j in range(max(1, len(self.tokens_fn_call) - 1)):
            self.fn_inputs.append(self.tokens_fn_call[j])

        # Handle last param to fn, might be a tag
        if len(self.tokens_fn_call) > 1:
            if self.tokens_fn_call[-1].find('<') == -1:
                self.fn_inputs.append(self.tokens_fn_call[-1])
            else:
                curr_tag = re.findall('<(.*?)>', self.tokens_fn_call[-1])[0]
                if curr_tag.startswith('tree'):
                    self.fn_inputs.append(self.branch_results[int(curr_tag[5:])])
                else:
                    if curr_tag.startswith('attr'):
                        self.fn_inputs.append(self.tag_instantiations[curr_tag]['type'])
                    self.fn_inputs.append(self.tag_instantiations[curr_tag]['value'])


    """
    Takes a 'iter_set_fn_list' branch and evaluates it self.set_size times, using the corresponding
    tag values.
    """
    def evaluate_set_branch(self, branch, set_op_input):
        set_iter_branch_result = []
        for i in range(self.set_size):
            branch_with_tag_idxs = [re.sub('{}', str(i+1), x) for x in branch]
            previous_result = set_op_input

            for i in range(len(branch_with_tag_idxs)):
                self.split_tokens_and_instantiate(branch_with_tag_idxs[i])
                self.fn_inputs += previous_result

                method_name = eval('self.' + self.fn_inputs[0])
                previous_result = [method_name(*self.fn_inputs[1:])]

            set_iter_branch_result.append(previous_result[0])

        return set_iter_branch_result


    """
    Takes a branch (an element from the template execution tree list) and evaluates it using the
    current instantiated tags.
    """
    def evaluate_branch(self, branch, input_all_objects=False, input_all_rooms=False):
        # Gather the necessary inputs to the execution branch
        inputs = branch[0].split('|')
        previous_result = []

        for inp in inputs:
            # Set of objects or rooms
            if inp.startswith('inputs_'):
                val = self.q_eval_info['inputs'][int(inp[-1])]
                if val in [ItemType.OBJECTS, ItemType.OBJECTS_REL]:
                    if input_all_objects:
                        previous_result.append(self.house['objects_all'])
                    else:
                        previous_result.append(self.house['objects'])
                elif not input_all_rooms:
                    previous_result.append(self.house['rooms'])
                else:
                    previous_result.append(self.house['rooms_all'])
            # Existing result from a previous execution branch
            elif inp.startswith('tree_'):
                previous_result.append(self.branch_results[int(inp[-1])])
            # Type (tag to be instantiated)
            else:
                assert inp[1:-1] in self.tag_instantiations, inp[1:-1]
                previous_result.append(self.tag_instantiations[inp[1:-1]]['value'])

        # Compute branch
        for i in range(1, len(branch)):
            self.split_tokens_and_instantiate(branch[i])
            # Need to pass result of the previous processing step
            self.fn_inputs += previous_result

            if self.tokens_fn_call[0] == 'iter_set':
                # Handle set(...) question template
                previous_result = self.evaluate_set_branch(self.q_eval_info['iter_set_fn_list'],
                                                           previous_result)
            else:
                # Call function
                method_name = eval('self.' + self.fn_inputs[0])
                previous_result = [method_name(*self.fn_inputs[1:])]

        self.branch_results.append(previous_result[0])


    """
    Renders trajectory in the house frame by frame and extracts all visited rooms and seen objects.
    """
    def generate_house_info(self):
        # Assumes self.traj_gen already generated and indexed seen objects and rooms
        self.house = { 'objects': [], 'objects_rel': [], 'rooms': [] }

        if ItemType.ROOMS in self.q_eval_info['inputs']:
            self.house['rooms'] = self.traj_gen.get_all_rooms()

        self.house['objects'] = self.traj_gen.get_all_objects()
        # Make sure that we have some knowledge of the room space.
        rooms_with_objects_seen = []
        for room in self.house['rooms']:
            room_found = False
            for obj in self.house['objects']:
                if obj['room_id'] == room['room_id']:
                    room_found = True
                    break
            if room_found:
                rooms_with_objects_seen.append(room)
        self.house['rooms'] = rooms_with_objects_seen

        if ItemType.OBJECTS_REL in self.q_eval_info['inputs']:
            self.house['objects_rel'] = self.traj_gen.get_all_nearby_object_pairs()
            self.house['objects'] = list(set([x[0] for x in self.house['objects_rel']['on']] +\
                                             [x[1] for x in self.house['objects_rel']['on']] +\
                                             [x[0] for x in self.house['objects_rel']['next_to']] +\
                                             [x[1] for x in self.house['objects_rel']['next_to']]))

            self.house['objects'] = [x.meta for x in self.house['objects']]
            for rel in ['on', 'next_to']:
                self.house['objects_rel'][rel] = [(x[0].meta, x[1].meta)
                                                  for x in self.house['objects_rel'][rel]]


    """
    Ensure if only one object has the room_location attribute, then the other one has 'everywhere'
    attached, to avoid asking ambiguous questions (e.g. "Are there more tables [everywhere] than
    rugs in the kitchen?")
    """
    def check_attributes_not_ambiguous(self):
        if 'room_location' in self.tag_instantiations['attr1']['type'] and\
           not 'room_location' in self.tag_instantiations['attr2']['type']:
            self.tag_instantiations['attr2']['type'].append('room_location')
            self.tag_instantiations['attr2']['value'].append('everywhere')
        elif 'room_location' in self.tag_instantiations['attr2']['type'] and\
             not 'room_location' in self.tag_instantiations['attr1']['type']:
            self.tag_instantiations['attr1']['type'].append('room_location')
            self.tag_instantiations['attr1']['value'].append('everywhere')


    """
    Check whether ground truth matches what has been seen on the trajectory for subset of questions
    which involve potentially multiple rooms of the same type (i.e. need to make sure we've seen all
    of them when asking to count/compare counts.)
    """
    def check_all_rooms_of_type_visited(self):
        if not self.q_template_string in [
            'Are there <comp> <room_type1-pl> than <room_type2-pl>?',
            'How many <room_type-pl> are there?']:
            return True

        if not 'rooms_all' in self.house:
            all_rooms_including_unseen = self.traj_gen.get_all_rooms(include_unseen_rooms=True)
            self.house['rooms_all'] = all_rooms_including_unseen

        # Check that all rooms of type have been visited
        for tag in self.tag_instantiations:
            if 'room_type' in tag:
                instantiated_room_type = self.tag_instantiations[tag]['value']
                rooms_on_trajectory = self.filter(
                    'room_type', instantiated_room_type, self.house['rooms'])
                rooms_in_house = self.filter(
                    'room_type', instantiated_room_type, self.house['rooms_all'])
                if len(rooms_on_trajectory) != len(rooms_in_house):
                    return False

        # Check that template evaluation gives same answer
        try:
            existing_eval_result = self.branch_results[-1]
            self.branch_results = []
            for branch in self.q_eval_info['tree']:
                self.evaluate_branch(branch, input_all_objects=False, input_all_rooms=True)
        except QuestionGenerationError as e:
            return False

        consistent_with_ground_truth = (self.branch_results[-1] == existing_eval_result)
        """
        if not consistent_with_ground_truth:
            print('Room type question answer not consistent with ground truth!')
        else:
            print('>>>SEEN ROOMS:', [x['room_id'] for x in self.house['rooms']])
            print('>>>ALL ROOMS:', [x['room_id'] for x in self.house['rooms_all']])
        """
        return consistent_with_ground_truth


    """
    Check whether more rooms have been seen for questions that ask about objects in the same room;
    this non-trivializes the question.
    """
    def check_more_rooms_seen(self):
        if not self.q_template_string in [
            'Are all <attr> <obj_type-pl> in the <room_type>?',
            'Are both the <attr1> <obj_type1> and the <attr2> <obj_type2> in the <room_type>?',
            'Are the <attr1> <obj_type1> and the <attr2> <obj_type2> in the same room?']:
            return True

        unique_obj_room_locations = list(set([x['room_id'] for x in self.house['objects']]))
        return len(unique_obj_room_locations) > 1


    """
    For questions that involve counting (e.g. "Are all chairs in the kitchen?"/"How many rugs are
    there in the bedroom?"/"Are there more televisions than computers in the office?"), run the
    template on all objects in the visited rooms, not only on the seen ones during the trajectory.
    This ensures that, even if some objects were not identified in the (approximate) matching of
    rendering and ground truth information, the answer is still correct.

    The method also runs additional checks to disambiguate meaning or avoid counting semantically
    ambiguous objects (i.e. kitchen cabinets).
    """
    def check_counts(self):
        question_involves_counting = False
        if 'all' in self.q_template_string:
            question_involves_counting = True
        else:
            for branch in self.q_eval_info['tree']:
                if 'count' in branch or 'count_unique' in branch:
                    question_involves_counting = True
                    break
        if not question_involves_counting:
            return True

        # Fix potentially ambiguous question by adding 'everywhere' attribute
        if self.q_template_string.startswith('Are there') and\
           ItemType.OBJECTS in self.q_eval_info['inputs']:
           self.check_attributes_not_ambiguous()
        # Don't allow kitchen_cabinets to be counted - ambiguous definition
        for tag in self.tag_instantiations:
            if 'obj_type' in tag and 'kitchen_cabinet' in self.tag_instantiations[tag]['value']:
                return False

        if not 'objects_all' in self.house:
            all_objects_including_unseen = self.traj_gen.get_all_objects(include_unseen_objects=True)
            self.house['objects_all'] = all_objects_including_unseen

        try:
            existing_eval_result = self.branch_results[-1]
            self.branch_results = []
            for branch in self.q_eval_info['tree']:
                self.evaluate_branch(branch, input_all_objects=True)
        except QuestionGenerationError as e:
            return False

        counts_consistent_with_ground_truth = (self.branch_results[-1] == existing_eval_result)
        """
        if not counts_consistent_with_ground_truth:
            print('Counts not consistent with ground truth!')
        """
        return counts_consistent_with_ground_truth


    """
    See banned_obj_* in constants.py. Ensures that we are not asking questions with low or zero
    entropy answers (e.g. "Where is the bed?"/"Are all toilets in the bathroom?").
    """
    def check_allowed_objects(self, question_answer):
        if self.q_template_string.startswith('Where '):
            assert type(question_answer) == str, 'Answer is not a str for Query<room_location>!'
        elif self.q_template_string in [
                'Are all <attr> <obj_type-pl> in the <room_type>?',
                'Are both the <attr1> <obj_type1> and the <attr2> <obj_type2> in the <room_type>?',
                'Is there set(<art> <attr{}> <obj_type{}>) in the <room_type>?',
                'Is there set(<art> <attr{}> <obj_type{}>)?',
                'Are the <attr1> <obj_type1> and the <attr2> <obj_type2> in the same room?',
                'Is there a room that has set(<art> <attr{}> <obj_type{}>)?']:
            assert type(question_answer) == bool, 'Answer is not a bool for exists question!'
        else:
            # Nothing to check here
            return True

        conflict_found = False
        for tag in self.tag_instantiations:
            if 'attr' in tag or not (self.tag_instantiations[tag]['value'] in
                                     banned_obj_room_combinations_existence_qs_negative or\
                                     self.tag_instantiations[tag]['value'] in
                                     banned_obj_entropy_sensitive_qs):
                continue
            # An object type on the forbidden lists is present, need to analyse whether the question
            # leaks too much information

            # If object type is on the banned list for the mentioned questions
            if self.tag_instantiations[tag]['value'] in banned_obj_entropy_sensitive_qs:
                conflict_found = True
                break
            # If question answer is negative and obj/room combination is forbidden
            elif question_answer == False and 'room_type' in self.tag_instantiations:
                # Get instantiated object type for current tag
                obj_type = self.tag_instantiations[tag]['value']
                forbidden_room_types_for_obj =\
                    banned_obj_room_combinations_existence_qs_negative[obj_type]

                # Find instantiated room type
                instantiated_room_type = self.tag_instantiations['room_type']['value']
                simple_room_types = instantiated_room_type.split('|')

                exists_non_conflicting_room = False
                for simple_room_type in simple_room_types:
                    if not simple_room_type in forbidden_room_types_for_obj:
                        exists_non_conflicting_room = True
                        break
                # Don't generate if negative answer and room/object combination is forbidden
                if not exists_non_conflicting_room:
                    conflict_found = True
                    break

        """
        if conflict_found:
            print('Object/room pair(s) not allowed!')
        """
        return not conflict_found


    """
    For <rel> questions it makes no sense to include the room location given twice, but only
    once and as an attribute of the second object, for a more natural-sounding question.
    (e.g. "Is the red thing next to the chair in the kitchen a table?" instead of
          "Is the red thing in the kitchen next to the chair in the kitchen a table?" or
          "Is the red thing in the kitchen next to the chair a table?")
    """
    def position_room_location_for_rel_questions(self):
        # Room location is given for both objects - leave it only for the 2nd object
        if 'room_location' in self.tag_instantiations['attr1']['type'] and\
           'room_location' in self.tag_instantiations['attr2']['type']:
           idx = self.tag_instantiations['attr1']['type'].index('room_location')
           self.tag_instantiations['attr1']['type'].remove('room_location')
           self.tag_instantiations['attr1']['value'].remove(
               self.tag_instantiations['attr1']['value'][idx])
        # Room location is given for the 1st object - transfer it to the 2nd one
        elif 'room_location' in self.tag_instantiations['attr1']['type']:
            idx = self.tag_instantiations['attr1']['type'].index('room_location')
            room_location = self.tag_instantiations['attr1']['value'][idx]
            self.tag_instantiations['attr2']['type'].append('room_location')
            self.tag_instantiations['attr2']['value'].append(room_location)
            self.tag_instantiations['attr1']['type'].remove('room_location')
            self.tag_instantiations['attr1']['value'].remove(room_location)


    """
    Given a question id, retrieves the question string and the rest of the associated info required
    for evaluating the corresponding template.
    """
    def load_question_info(self, question_id):
        keys = list(sorted(list(self.q_templates_eval_nodes.keys())))

        # Find the question
        self.q_template_string = keys[question_id]
        self.q_eval_info = self.q_templates_eval_nodes[self.q_template_string]
        self.check_fine_grained_room_counts()


    """
    Given the answer obtained through evaluating the question template tree with the current tag
    instantiations, check if it matches the one required.
    """
    def check_answer(self):
        return self.branch_results[-1] == self.answer or type(self.answer) == str


    """
    Generates the currently instantiated question with the provided answer, using the current
    trajectory.
    """
    def generate_question(self):
        # Gather tags that require instantiation
        tags_init = re.findall('<(.*?)>', self.q_template_string)
        tags = []

        # Handle tags belonging to a set of templated items
        if 'set_size' in self.q_eval_info:
            self.set_size = self.q_eval_info['set_size']()
        for tag in tags_init:
            if '{}' in tag:
                for i in range(self.set_size):
                    tag_with_id = re.sub('{}', str(i+1), tag)
                    tags.append(tag_with_id)
            else:
                tags.append(tag)

        # Extract types from the tags
        types = [self.strip_nums(self.strip_pl(self.strip_idx_placeholder(x))) for x in tags]
        for i in range(len(types)):
            if not types[i] in self.tags_to_instantiate:
                types[i] = None

        # tag name -> tag instantiated value, branch index
        self.tag_instantiations = {}
        for i in range(len(tags)):
            if types[i] != None:
                self.tag_instantiations[self.strip_pl(tags[i])] = {
                    'value': None,
                    'type': types[i],
                }

        # Get room and object info
        self.generate_house_info()

        # Execute functional tree and choose values for tags
        self.branch_results = []
        valid_instantiation = False

        # start = time.time()
        while not valid_instantiation:
            exception = False
            try:
                self.get_next_instantiation()

                self.branch_results = []
                for branch in self.q_eval_info['tree']:
                    self.evaluate_branch(branch)
            except QuestionGenerationError as e:
                exception = True

            # Implement a few checks which are question-dependent
            if not exception and\
               self.check_answer() and\
               self.check_counts() and\
               self.check_all_rooms_of_type_visited() and\
               self.check_more_rooms_seen() and\
               self.check_allowed_objects(self.branch_results[-1]):
                valid_instantiation = True

        self.answer = self.branch_results[-1]
        # print('Result of tree evaluation:', self.answer, 'in', time.time() - start, 'seconds.')


        if '<rel>' in self.q_template_string:
            self.position_room_location_for_rel_questions()

        q_build = QuestionBuild(self.q_template_string)
        q_string = q_build.replace_tags_with_values(tag_instantiations=self.tag_instantiations,
                                                    set_size=self.set_size)

        example = (q_string, self.answer)
        return example


    """
    Updates the trajectory that we want to render and process.
    """
    def update_trajectory(self, traj_id):
        self.traj_gen.update_trajectory(traj_id)


    """
    Renders a trajectory and saves all the frames.
    """
    def generate_trajectory_and_seen_items(self):
        self.traj_gen.generate_trajectory_and_seen_items(frame_dir=None)


    """
    Generates a dataset example with the currently instantiated trajectory, trying to have the given
    question id and answer.
    """
    def generate_example(self, question_id, answer):
        curr_q_id = question_id
        self.load_question_info(curr_q_id)
        self.answer = answer

        return self.generate_question()
