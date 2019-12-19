import argparse
import collections
import logging
import pickle

import pandas
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from model.definition import *
from tokenization import BertTokenizer
from utils.dataset_prepare_utils import convert_example_to_features

logger = logging.getLogger(__name__)


class GlossBERTDataset(Dataset):
    def __init__(self, data, tokenizer, **kwargs):
        # dataset content
        self._tokenizer = tokenizer
        self._sentences = []

        word_to_senses = {}
        last_sent_id = ''
        last_target_id = ''
        for i, item in enumerate(tqdm(data, desc="CSV Line Iteration"), start=0):
            target_id, label, text, gloss, target_index_start, target_index_end, sense_key = item
            doc_id, sent_id, inst_id = target_id.split('.')
            if doc_id + sent_id != last_sent_id:
                last_sent_id = doc_id + sent_id
                sentence = Sentence('.'.join((doc_id, sent_id)), text)
                self.add_sentence(sentence)

            orig_tokens = text.split(' ')
            if target_id != last_target_id:
                bert_tokens = []
                bert_tokens.append("[CLS]")
                for length in range(len(orig_tokens)):
                    if length == target_index_start:
                        target_to_tok_map_start = len(bert_tokens)
                    if length == target_index_end:
                        target_to_tok_map_end = len(bert_tokens)
                        break
                    bert_tokens.extend(tokenizer.tokenize(orig_tokens[length]))
                target_word = ' '.join(orig_tokens[target_index_start:target_index_end])
                if target_word not in word_to_senses.keys():
                    word_to_senses[target_word] = [sense_key]
                elif sense_key not in word_to_senses[target_word]:
                    word_to_senses[target_word].append(sense_key)
                last_target_id = target_id
                instance = Instance(target_id, sentence, target_word,
                                    start_pos=target_to_tok_map_start, end_pos=target_to_tok_map_end)
                sentence.add_instance(instance)
                instance.add_candidate_sense(sense_key, gloss, label)
            else:
                if target_word not in word_to_senses.keys():
                    word_to_senses[target_word] = [sense_key]
                elif sense_key not in word_to_senses[target_word]:
                    word_to_senses[target_word].append(sense_key)
                instance.add_candidate_sense(sense_key, gloss, label)
        sense_freq_counter = collections.Counter([item[6] if item[1] == 1 else '' for item in data])
        self.set_word_to_senses(word_to_senses)
        self.set_sense_freq_counter(sense_freq_counter)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def add_sentence(self, sentence):
        self._sentences.append(sentence)

    def filter_data(self, max_seq_length):
        pass

    def set_word_to_senses(self, word_to_senses):
        self.word_to_senses = word_to_senses

    def set_sense_freq_counter(self, sense_freq_counter):
        self.sense_freq_counter = sense_freq_counter

    @classmethod
    def from_data_csv(cls, data_csv_path, tokenizer, **kwargs):
        """
        csv file format:
        target_id	label	sentence	gloss	target_index_start	target_index_end	sense_key
        ['d000.s000.t000', 0, 'How long has it been since you reviewed the objectives of your benefit and service program ?', 'desire strongly or persistently', 1, 2, 'long%2:37:02::']
        :param data_csv_path: the path of csv file containing data.
        :return: the GlossBERTDataset instance
        """
        data = pandas.read_csv(data_csv_path, sep="\t", na_filter=False).values
        dataset = cls(data, tokenizer, **kwargs)

        return dataset


class GlossBERTDataset_for_CGPair_Feature(GlossBERTDataset):
    def __init__(self, data, tokenizer, **kwargs):
        super().__init__(data, tokenizer, **kwargs)
        try:
            self.max_seq_length = kwargs['max_seq_length']
        except:
            self.max_seq_length = 100
        #self.positive_examples = []
        #self.negative_examples = []

        self.pos_indexes = []
        self.neg_indexes = []
        self.all_examples = []
        self.num_invalid_indexes = []
        for sentence in tqdm(self._sentences, desc="Sentence Iteration"):
            for instance in sentence:
                for idx, cand_sense in enumerate(instance, start=0):
                    sense_key, gloss, label = cand_sense
                    if instance.end_pos >= self.max_seq_length:
                        label = -1
                    # sample: ((self.sentence.text, gloss), is_next, start_pos, span_length)
                    cur_example = InputExample(guid=instance.id, text_a=sentence.text,
                                 start_id=instance.start_pos, end_id=instance.end_pos,
                                 text_b=gloss, label=label)
                    if label == 1:
                        self.pos_indexes.append(len(self.all_examples))
                        #self.positive_examples.append(cur_example)
                    elif label == 0:
                        self.neg_indexes.append(len(self.all_examples))
                        #self.negative_examples.append(cur_example)
                    else:  # label == -1
                        self.num_invalid_indexes.append(len(self.all_examples))
                    self.all_examples.append(cur_example)

        self.all_features = []
        for example in tqdm(self.all_examples, desc="Training Example Iteration"):
            feature = convert_example_to_features(example, self.max_seq_length, tokenizer)
            self.all_features.append(feature)

    def __getitem__(self, item):
        return self.all_features[item]

    def __len__(self):
        assert len(self.all_examples) == len(self.all_features)
        return len(self.all_features)


class GlossBERTDataset_for_Sentence(GlossBERTDataset):
    def __init__(self, data, tokenizer, **kwargs):
        super().__init__(data, tokenizer, **kwargs)

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, item):
        return self._sentences[item]


def _parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--target",
                        default='train',
                        type=str,
                        choices=['train', 'dev', 'semeval2013', 'semeval2015', 'semeval2', 'semeval3'])
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = _parse_args()

    csv_paths = {
        'train':       '../Training_Corpora/SemCor/semcor_train_token_cls.csv',
        'dev':         '../Evaluation_Datasets/semeval2007/semeval2007_test_token_cls.csv',
        'seneval2013': '../Evaluation_Datasets/semeval2013/semeval2013.csv',
        'seneval2015': '../Evaluation_Datasets/semeval2015/semeval2015.csv',
        'seneval2':    '../Evaluation_Datasets/senseval2/senseval2.csv',
        'seneval3':    '../Evaluation_Datasets/senseval3/senseval3.csv',
        'ALL':         '../Evaluation_Datasets/ALL/ALL.csv'
    }
    tokenizer = BertTokenizer.from_pretrained('../bert-model', do_lower_case=True)

    #with open('../Training_Corpora/SemCor/train_glossbert_dataset.pkl', 'rb') as rbf:
        #glossbert_dataset = pickle.load(rbf)

    target = 'dev'
    if target == 'train':
        glossbert_dataset = GlossBERTDataset_for_CGPair_Feature.from_data_csv(
            csv_paths['train'], tokenizer, max_seq_length=args.max_seq_length)
        with open('../Training_Corpora/SemCor/train_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
    elif target == 'dev':
        glossbert_dataset = GlossBERTDataset_for_CGPair_Feature.from_data_csv(
            csv_paths['dev'], tokenizer, max_seq_length=args.max_seq_length)
        with open('../Evaluation_Datasets/semeval2007/semval2007_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
    elif target == 'semeval2013':
        glossbert_dataset = GlossBERTDataset_for_Sentence.from_data_csv(
            csv_paths['semeval2013'], tokenizer, max_seq_length=args.max_seq_length)
        with open('../Evaluation_Datasets/semeval2013/semval2013_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
    elif target == 'semeval2015':
        glossbert_dataset = GlossBERTDataset_for_Sentence.from_data_csv(
            csv_paths['semeval2015'], tokenizer, max_seq_length=args.max_seq_length)
        with open('../Evaluation_Datasets/semeval2015/semval2015_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
    elif target == 'semeval2':
        glossbert_dataset = GlossBERTDataset_for_Sentence.from_data_csv(
            csv_paths['semeval2'], tokenizer, max_seq_length=args.max_seq_length)
        with open('../Evaluation_Datasets/semeval2/semval2_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
    elif target == 'semeval3':
        glossbert_dataset = GlossBERTDataset_for_Sentence.from_data_csv(
            csv_paths['semeval3'], tokenizer, max_seq_length=args.max_seq_length)
        with open('../Evaluation_Datasets/semeval3/semval3_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
