import random

from torch.utils.data import Sampler

from dataset.glossbert_dataset import GlossBERTDataset_for_CGPair_Feature


class NegDownSampler(Sampler):

    def __init__(self, data_source, neg_pos_ratio=None):
        super().__init__(data_source)
        if not isinstance(data_source, GlossBERTDataset_for_CGPair_Feature):
            raise ValueError("data_source should be a dataset of type GlossBERTDataset_for_CGPair_Feature, but got "
                             "data_source={}".format(data_source))
        self.pos_indexes = data_source.pos_indexes
        self.neg_indexes = data_source.neg_indexes
        self.default_neg_pos_ratio = int(len(self.neg_indexes)/len(self.pos_indexes)) + 1
        if neg_pos_ratio is None:
            self.neg_pos_ratio = self.default_neg_pos_ratio
        else:
            self.neg_pos_ratio = neg_pos_ratio
        self.__resample_neg_instance()
        # self.num_pos_samples = len(data_source.pos_indexes)
        # self.num_neg_samples = len(data_source.neg_indexes)
        # self.index_mapping = data_source.index_mapping

    def __iter__(self):
        return iter(self.selected_indexes)

    def __len__(self):
        return len(self.selected_indexes)

    def refresh_sampler(self, neg_pos_ratio=None):
        if neg_pos_ratio is not None:
            self.neg_pos_ratio = neg_pos_ratio
        self.__resample_neg_instance(neg_pos_ratio=self.neg_pos_ratio)

    def __resample_neg_instance(self):
        num_pos_indexes = len(self.pos_indexes)
        num_neg_indexes = len(self.neg_indexes)

        n_neg = min(num_neg_indexes, int(num_pos_indexes*max(0, self.neg_pos_ratio)))
        neg_rand_n_list = random.sample(range(num_neg_indexes), n_neg)

        self.selected_indexes = self.pos_indexes + [self.neg_indexes[i] for i in neg_rand_n_list]
        random.shuffle(self.selected_indexes)