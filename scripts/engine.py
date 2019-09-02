import json
import numpy as np
import os
import random
import signal

from question_gen import QuestionGenerator
from trajectory_gen import TrajectoryGenerator

# Numeric constants
TIMEOUT = 0.5
TIMEOUT_REL = 0.8
MAX_COUNT_ANSWER = 10


class QuestionEngine():

    def __init__(self, question_generator, save_dir):
        self.question_generator = question_generator
        self.save_dir = save_dir

        # Store answer type/distribution for each question and how many have been generated so far
        self.question_set = {}
        for q_string in self.question_generator.q_templates_eval_nodes:
            self.question_set[q_string] = {}

            ans_type = self.question_generator.q_templates_eval_nodes[q_string]['ans_type']
            self.question_set[q_string]['ans_type'] = ans_type
            self.question_set[q_string]['generated_count'] = 0

            self.question_set[q_string]['ans_distribution'] = {}
            if ans_type == bool:
                self.question_set[q_string]['ans_distribution'][False] = 0
                self.question_set[q_string]['ans_distribution'][True] = 0
            elif ans_type == int:
                for i in range(1, MAX_COUNT_ANSWER + 1):
                    self.question_set[q_string]['ans_distribution'][i] = 0
            else:
                # Other kinds of answers depend a lot on the context of each trajectory
                self.question_set[q_string]['ans_distribution'] = {}

        # Keep question template strings separately and sorted for fast indexing during generation
        self.q_templates = list(sorted(list(self.question_set.keys())))

        # Set to False upon failing to generate a question
        self.question_generated = False
        # Queue containing the ids of questions that could not be generated
        self.generate_later_queue = []

        # Container for the generated examples
        self.dataset = {}
        # Container for generated question-answer pairs, don't want duplicates
        self.generated_pairs = {}

        print('Question engine init done!')


    """
    Handler for question generation timeout (could not generate a question of the current type).
    """
    def could_not_generate(self, signum, frame):
        self.question_generated = False
        raise Exception('Could not generate before timeout!')


    """
    Write example generated and update question/answer distribution, unless they are duplicates.
    """
    def write_example(self, q_id, q_text, q_answer, q_template):
        print('### Success!', q_text, q_answer)

        key = q_text + '|' + str(q_answer)
        if key in self.generated_pairs:
            print('Not saving example; question-answer pair already existent')
            return
        else:
            self.generated_pairs[key] = 1

        # In case of a <query>:... question, the answer might not have come up before
        if not q_answer in self.question_set[q_template]['ans_distribution']:
            self.question_set[q_template]['ans_distribution'][q_answer] = 0

        # Update distribution
        self.question_set[q_template]['ans_distribution'][q_answer] += 1
        self.question_set[q_template]['generated_count'] += 1

        # Write example to file
        traj_id = self.question_generator.traj_gen.traj_id
        self.dataset[traj_id] = {
            'q_id': q_id,
            'tag_instantiation': self.question_generator.tag_instantiations,
            'q_text': q_text,
            'q_ans': q_answer,
        }


    """
    Writes to file the data that was generated.
    """
    def dump_dataset(self):
        if not self.dataset:
            return

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        filename = self.question_generator.traj_gen.house_id + '.json'
        path = os.path.join(self.save_dir, filename)

        with open(path, 'w') as out_file:
            json.dump(self.dataset, out_file)
        self.dataset.clear()


    """
    Generate a question for the current trajectory; if there is no generated example before TIMEOUT
    seconds, try generating another question/answer pair.
    """
    def generate_with_timeout(self, traj_id):
        # First update with current trajectory
        self.question_generator.update_trajectory(traj_id)

        # Render frames and extract information
        self.question_generator.generate_trajectory_and_seen_items()
        print('Done rendering trajectory!')

        self.question_generated = False
        q_ids_attempted = []
        # Attempt to generate a question for the current trajectory
        while not self.question_generated:
            # Pick a question, either from previously non-generated ones or at random
            q_id = None
            if len(self.generate_later_queue) > 0 and\
               self.generate_later_queue[0][1] != traj_id:
                # This should not be previously ungenerated for the _current_ trajectory!
                q_id, _ = self.generate_later_queue[0]
                self.generate_later_queue = self.generate_later_queue[1:]
            else:
                q_id = random.randint(0, len(self.q_templates) - 1)

            q_template = self.q_templates[q_id]
            print('Trying to generate question with id', q_id)
            q_ids_attempted.append(q_id)

            # Decide, if possible, what answer we want the question to have
            ans_type = self.question_set[q_template]['ans_type']
            ans_distribution = self.question_set[q_template]['ans_distribution']
            required_answers = []

            # Yes, no
            if ans_type == bool:
                if ans_distribution[False] < ans_distribution[True]:
                    required_answers.append(False)
                else:
                    required_answers.append(True)

            # Count
            elif ans_type == int:
                q_counts = []
                for i in range(1, MAX_COUNT_ANSWER + 1):
                    q_counts.append(ans_distribution[i])
                # Add 1 to the required answers because count answers start from 1, not 0
                required_answers = list(np.argsort(np.array(q_counts)) + 1)

            # Other (color, obj_type, room_location, room_type)
            else:
                # Nothing to do here
                required_answers.append('<query>:' + ans_type)

            for required_ans in required_answers:
                self.question_generated = True
                # print('Trying to generate question with id', q_id, 'and answer', required_ans)

                # Register timeout handler
                timeout = TIMEOUT
                if q_id in [15, 24, 26, 29]:
                    timeout = TIMEOUT_REL
                signal.signal(signal.SIGALRM, self.could_not_generate)
                signal.setitimer(signal.ITIMER_REAL, timeout)

                try:
                    (q_text, q_answer) = self.question_generator.generate_example(q_id,
                                                                                  required_ans)
                except Exception as e:
                    print(e)
                signal.setitimer(signal.ITIMER_REAL, 0)

                # Generation was successful
                if self.question_generated:
                    if type(required_ans) != str:
                        assert q_answer == required_ans
                    self.write_example(q_id, q_text, q_answer, q_template)
                    return

            if not self.question_generated:
                # Add question to the queue, will try to generate it for another trajectory
                if not (q_id, traj_id) in self.generate_later_queue:
                    self.generate_later_queue.append((q_id, traj_id))
                    # print(self.generate_later_queue)
                # All questions have been attempted unsuccessfully - discard the trajectory
                if set(q_ids_attempted) == set(range(0, len(self.q_templates))):
                    print('!!! Could not generate a question for trajectory', traj_id)
                    return
