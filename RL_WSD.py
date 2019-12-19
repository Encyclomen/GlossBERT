# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict
import csv
import logging
import os
import random
import pandas as pd
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

from dataset.sampler import NegDownSampler
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from model.definition import InputExample, InputFeatures
from model.modeling import *
from model.base_model import BaseModel
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from dataset.glossbert_dataset import *

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--mode",
                        default='bert_pretrain',
                        type=str,
                        choices=["bert_pretrain", "bert_test"],
                        help="The mode to run.")
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--label_data_dir",
                        default=None,
                        type=str,
                        help="The label data dir. (./wordnet)")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help='''a path or url to a pretrained model archive containing:
                            'bert_config.json' a configuration file for the model
                            'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--neg_pos_ratio",
                        default=1,
                        type=float,
                        help="Ratio of negative training examples over positive ones.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    return args


def bert_pretrain(model, dataset):
    #sampler = SequentialSampler(dataset)
    #sampler = RandomSampler(glossbert_dataset)
    sampler = NegDownSampler(dataset, neg_pos_ratio=args.neg_pos_ratio)
    bert_pretrain_dataloader = DataLoader(dataset, sampler=sampler, batch_size=1,
                                          collate_fn=lambda x: zip(*x))
    num_train_optimization_steps = int(
        len(sampler) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    observe_interval = 1000
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    # Assign loss function
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        wf = open(('output/log_%d.txt' % epoch), 'w')
        tr_loss = 0
        for step, batch in enumerate(tqdm(bert_pretrain_dataloader, desc="Iteration"), start=1):
            guid, input_ids1, input_mask1, segment_ids1, \
            input_ids2, input_mask2, segment_ids2, \
            start_id, end_id, label = batch

            input_id1s_tensor = torch.tensor(input_ids1, dtype=torch.long, device=device)
            input_mask1_tensor = torch.tensor(input_mask1, dtype=torch.long, device=device)
            segment_ids1_tensor = torch.tensor(segment_ids1, dtype=torch.long, device=device)
            input_id2s_tensor = torch.tensor(input_ids1, dtype=torch.long, device=device)
            input_mask2_tensor = torch.tensor(input_mask1, dtype=torch.long, device=device)
            segment_ids2_tensor = torch.tensor(segment_ids1, dtype=torch.long, device=device)
            label_tensor = torch.tensor(label, dtype=torch.long, device=device)

            batch_size, seq_len = input_id1s_tensor.size()
            # The mask has 1 for real target
            selection_mask_tensor = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                selection_mask_tensor[i][start_id[i]:end_id[i]] = 1

            logits = model(input_id1s_tensor, input_mask1_tensor, segment_ids1_tensor, selection_mask_tensor,
                           input_id2s_tensor, input_mask2_tensor, segment_ids2_tensor)
            loss = loss_function(logits, label_tensor)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if (step % args.gradient_accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step % observe_interval == 0:
                print('Epoch: %d, Step: %d, avg_loss: %.2f' % (epoch, step, (tr_loss/observe_interval)))
                wf.write('Epoch: %d, Step: %d, avg_loss: %.4f\n' % (epoch, step, (tr_loss / observe_interval)))
                tr_loss = 0
        wf.close()
        # Save a trained model
        logger.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, '_'.join((str(epoch), WEIGHTS_NAME)))
        torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':
    bert_pretrain_logger = logging.getLogger(__name__)

    args = parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    bert_pretrain_logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    tokenizer = BertTokenizer.from_pretrained('bert-model', do_lower_case=True)
    if args.mode == 'bert_pretrain':

        glossbert_dataset = GlossBERTDataset_for_CGPair_Feature.from_data_csv(
            'Training_Corpora/SemCor/semcor_train_token_cls.csv', tokenizer, max_seq_length=args.max_seq_length)
        with open('Training_Corpora/SemCor/train_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
        #with open('Training_Corpora/SemCor/train_glossbert_dataset.pkl', 'rb') as rbf:
            #glossbert_dataset = pickle.load(rbf)
        # Load open-source bert
        bert_model = BertModel.from_pretrained('bert-model')
        model = BaseModel(bert_model).to(device)
        bert_pretrain(model, glossbert_dataset)
    elif args.mode == 'bert_test':
        with open('Evaluation_Datasets/semeval2013/semval2013_glossbert_dataset.pkl', 'rb') as rbf:
            glossbert_dataset = pickle.load(rbf)
        # Load open-source bert
        bert_model = BertModel.from_pretrained('.cache')
        model = BaseModel(bert_model).to(device)
        model.load_state_dict(torch.load('output/2_pytorch_model.bin', map_location=torch.device('cpu')))
        bert_pretrain(model, glossbert_dataset)