class Sentence:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.instances = []

    def add_instance(self, instance):
        self.instances.append(instance)

    def __iter__(self):
        return iter(self.instances)

    def get_n_instance(self):
        return len(self.instances)

    def get_total_cand_senses(self):
        return sum([instance.get_n_cand_senses() for instance in self.instances])


class Instance:
    #counter = 0
    #counter2 = 0
    def __init__(self, id, sentence, target_word, start_pos, end_pos):
        self.id = id
        self.sentence = sentence
        self.target_word = target_word
        self.start_pos = int(start_pos)
        self.end_pos = int(end_pos)

        self.candidate_senses = []

    def add_candidate_sense(self, sense_key, gloss, label):
        self.candidate_senses.append((sense_key, gloss, label))

    def __iter__(self):
        return iter(self.candidate_senses)

    def get_n_cand_senses(self):
        return len(self.candidate_senses)

    def get_gold_label_index(self):
        for idx, cand_sense in enumerate(self.candidate_senses, start=0):
            if cand_sense[-1] == 1:
                return idx
        return -1


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, start_id, end_id, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = int(label)
        self.start_id = int(start_id)
        self.end_id = int(end_id)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, target_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.target_mask = target_mask
