import re


class QuestionBuild():

    def __init__(self, q_text_template):
        self.q_text_template = q_text_template

    """
    General types correspond to room, object and color tags.
    """
    def replace_general_types(self, items):
        for tag in items:
            replace_string = items[tag]

            # Pluralise
            if tag[:-1] + '-pl' in self.q_text_template:
                tag = tag[:-1] + '-pl>'

                if replace_string == 'switch':
                    replace_string += 'es'
                elif replace_string == 'balcony':
                    replace_string = replace_string[:-1] + 'ies'
                elif replace_string != 'shoes':
                    replace_string += 's'

            # Tags should be unique from the question template descriptions in QuestionGenerator
            self.q_text_template = self.q_text_template.replace(tag, replace_string)

        self.q_text_template = re.sub(' +',' ', self.q_text_template)


    """
    Attributes describe items (rooms and objects).
    """
    def replace_item_attributes(self, attr_info):
        attr_tags = list(attr_info.keys())

        for i in range(len(attr_tags)):
            tag = attr_tags[i]
            attr_types = attr_info[tag]['type']

            for j in range(len(attr_info[tag]['value'])):
                replace_string = attr_info[tag]['value'][j]

                if attr_types[j] != 'room_location':
                    # Insert attribute before the word
                    pos = self.q_text_template.find(tag)
                    self.q_text_template = self.q_text_template[:pos] + replace_string +\
                                           self.q_text_template[pos:]
                else:
                    # Room location needs to go after the word
                    pos = self.q_text_template.find(tag)
                    toks = self.q_text_template[pos:].split(' ')

                    if len(toks) < 2:
                        # Question is of the form '...<attr>?'
                        insert_pos = len(self.q_text_template) - 1
                    else:
                        insert_pos = pos + self.q_text_template[pos:].find(toks[1]) + len(toks[1])

                    if self.q_text_template[insert_pos - 1] == '?':
                        insert_pos -= 1

                    if replace_string != 'everywhere':
                        replace_string = 'located in the ' + replace_string
                    self.q_text_template = self.q_text_template[:insert_pos] + ' ' +\
                                           replace_string + self.q_text_template[insert_pos:]

            self.q_text_template = self.q_text_template.replace(tag, '')
            # Special case for 'How many <obj_type-pl> are <attr>?' with <attr> having no values
            if self.q_text_template.endswith('are ?'):
                self.q_text_template = self.q_text_template[:-1] + 'there?'

        self.q_text_template = re.sub(' +',' ', self.q_text_template)


    """
    Articles are 'a', 'an'. The next words should not be tags.
    """
    def replace_articles(self):
        pos = self.q_text_template.find('<art>')
        while pos != -1:
            assert pos + 6 < len(self.q_text_template), 'Beyond end of question string!'
            first_letter = self.q_text_template[pos + 6]
            if first_letter in 'aeiou':
                replace_string = 'an'
            else:
                replace_string = 'a'
            self.q_text_template = self.q_text_template.replace('<art>', replace_string, 1)
            pos = self.q_text_template.find('<art>')
        self.q_text_template = re.sub(' +',' ', self.q_text_template)


    """
    Special case for a set of items - template needs to be expanded dynamically, based on the count.
    """
    def expand_with_set(self, set_size):
        # Find start position of set tags
        set_tags = re.findall('set\((.*?)\)', self.q_text_template)[0]
        pos = self.q_text_template.find('set(')

        # Replicate tags and assign indices
        expanded = ''
        for i in range(set_size - 1):
            expanded += re.sub('{}', str(i+1), set_tags) + ' and '
        expanded += re.sub('{}', str(set_size), set_tags)

        # Construct expanded question template
        self.q_text_template = self.q_text_template[:pos] + expanded +\
                               self.q_text_template[pos + 5 + len(set_tags):]
        self.q_text_template = re.sub(' +',' ', self.q_text_template)


    """
    Wrap tag name around '<...>'.
    """
    @staticmethod
    def tagify(tag_instantiations):
        new_instantiations = {}
        for tag in tag_instantiations:
            new_instantiations['<' + tag + '>'] = tag_instantiations[tag]
        return new_instantiations


    """
    Make all the possible substitutions, according to the tag instantiations received.
    """
    def replace_tags_with_values(self, tag_instantiations, set_size=None):
        tag_instantiations = QuestionBuild.tagify(tag_instantiations)

        if 'set(' in self.q_text_template:
            assert set_size != None, 'Did not receive a set size for the question!'
            self.expand_with_set(set_size)

        # For <attr> tags
        attr_info = {}
        # For <room_type>, <obj_type>, <color> tags
        other = {}

        for tag in tag_instantiations:
            if tag in ['<rel>', '<comp>', '<comp_rel>', '<comp_sup>']:
                self.q_text_template = self.q_text_template.replace(
                    tag, tag_instantiations[tag]['value'])
            elif 'attr' in tag:
                assert len(tag_instantiations[tag]['value']) ==\
                       len(tag_instantiations[tag]['type']),\
                       '\'value\' and \'type\' list sizes for attr tags don\'t match!'
                attr_info[tag] = tag_instantiations[tag]
            else:
                other[tag] = tag_instantiations[tag]['value']

        self.replace_general_types(other)
        self.replace_item_attributes(attr_info)
        self.replace_articles()
        # Replace '_'s in object/room names with spaces
        self.q_text_template = self.q_text_template.replace('_', ' ')
        # Replace '|'s in composite room type names with '/'s
        self.q_text_template = self.q_text_template.replace('|', '/')

        return self.q_text_template
