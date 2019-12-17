import pickle

import torch
from torch.utils.data import BatchSampler, RandomSampler, DataLoader, SequentialSampler

from dataset.glossbert_dataset import GlossBERTDataset_for_CGPair_Feature
from tokenization import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-model', do_lower_case=True)
glossbert_dataset = GlossBERTDataset_for_CGPair_Feature.from_data_csv(
    'Evaluation_Datasets/semeval2007/semeval2007_test_token_cls.csv', tokenizer)
#batch_sampler = BatchSampler(SequentialSampler(glossbert_dataset), batch_size=3, drop_last=True)
sequential_sampler = SequentialSampler(glossbert_dataset)
bert_pretrain_dataloader = DataLoader(glossbert_dataset, sampler=sequential_sampler, batch_size=3, collate_fn=lambda x: zip(*x))
for step, batch in enumerate(bert_pretrain_dataloader, start=1):
    guid, input_ids1, input_mask1, segment_ids1, \
    input_ids2, input_mask2, segment_ids2, \
    start_id, end_id, label = batch

    input_id1s_tensor = torch.tensor(input_ids1, dtype=torch.long)
    input_mask1_tensor = torch.tensor(input_mask1, dtype=torch.long)
    segment_ids1_tensor = torch.tensor(segment_ids1, dtype=torch.long)
    input_id2s_tensor = torch.tensor(input_ids1, dtype=torch.long)
    input_mask2_tensor = torch.tensor(input_mask1, dtype=torch.long)
    segment_ids2_tensor = torch.tensor(segment_ids1, dtype=torch.long)
    label_tensor = torch.tensor(label, dtype=torch.long)
    b = list(zip(*batch))
    a=1
a=1