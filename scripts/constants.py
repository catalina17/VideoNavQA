# From House3D/metadata/colormap_coarse.csv
query_objects = {
    'bathtub' : 1,
    'bed' : 1,
    'chair' : 1,
    'clock' : 1,
    'computer' : 1,
    'curtain' : 1,
    'desk' : 1,
    'door' : 1,
    'dresser' : 1,
    'fan' : 1,
    'gym_equipment' : 1,
    'hanging_kitchen_cabinet' : 1,
    'heater' : 1,
    'kitchen_cabinet' : 1,
    'mirror' : 1,
    'ottoman' : 1,
    'pillow' : 1,
    'rug' : 1,
    'sofa' : 1,
    'shoes' : 1,
    'shower' : 1,
    'sink' : 1,
    'stand' : 1,
    'switch' : 1,
    'table' : 1,
    'television' : 1,
    'toilet' : 1,
    'trash_can' : 1,
    'tv_stand' : 1,
    'vase' : 1,
    'vehicle' : 1,
    'wardrobe_cabinet' : 1,
}

exclude_rooms = {
    'room': 1,
    '': 1,
}

"""
For the following questions:
    - equals: "Are all <attr> <obj_type-pl> in the <room_type>?"
    - exists: "Is there set(<art> <attr{}> <obj_type{}>) in the <room_type>?",
              "Are the <attr1> <obj_type1> and the <attr2> <obj_type2> in the <room_type>?"
    - query(<room_location>): "Where are/is the...?"
The negative combinations are for the equals/exists questions (e.g. there can not be a shower in the
bedroom).
"""
# Objects that only have one very likely location
banned_obj_entropy_sensitive_qs = [
    'bathtub',
    'bed',
    'hanging_kitchen_cabinet',
    'kitchen_cabinet',
    'shower',
    'toilet',
    'vehicle',
]
# Objects mapped to locations that are not sensible
banned_obj_room_combinations_existence_qs_negative = {
    'computer' : ['balcony', 'bathroom', 'boiler_room', 'garage', 'loggia', 'terrace', 'wardrobe'],
    'desk' : ['bathroom', 'wardrobe'],
    'dresser' : ['balcony', 'bathroom', 'boiler_room', 'garage', 'gym', 'kitchen', 'loggia',
                 'terrace'],
    'gym_equipment' : ['bathroom', 'dining_room'],
    'ottoman' : ['garage'],
    'pillow' : ['bathroom', 'boiler_room', 'garage'],
    'sofa' : ['bathroom', 'garage'],
    'sink' : ['balcony', 'bedroom', 'child_room', 'dining_room', 'entryway', 'hall', 'hallway',
              'living_room', 'lobby', 'loggia', 'office', 'terrace', 'wardrobe'],
    'television' : ['bathroom', 'boiler_room', 'wardrobe'],
    'tv_stand' : ['bathroom', 'boiler_room', 'wardrobe'],
    'wardrobe_cabinet' : ['balcony', 'kitchen', 'loggia', 'terrace'],
}

all_simple_room_types = [
    'balcony',
    'bathroom',
    'bedroom',
    'boiler_room',
    'child_room',
    'dining_room',
    'entryway',
    'garage',
    'gym',
    'hall',
    'hallway',
    'kitchen',
    'living_room',
    'lobby',
    'loggia',
    'office',
    'storage',
    'terrace',
    'wardrobe',
]
