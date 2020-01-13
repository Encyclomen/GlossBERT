# coding=utf-8
from __future__ import absolute_import, division, print_function

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler)
from tqdm import trange

from dataset.sampler import NegDownSampler
from model.agent import Agent
from model.modeling import *
from model.base_model import BaseModel2 as BaseModel
from optimization import BertAdam
from dataset.glossbert_dataset import *

logger = logging.getLogger(__name__)


def compute_reward(new_logits, ori_logits, label_tensor, sep_pos, num_instances, sample_probs, selected_instance_idx, alpha=0.5):
    new_sense_wise_cross_entropy = F.cross_entropy(new_logits, label_tensor, weight=torch.tensor([1, 8], dtype=torch.float, device=device), reduction='none')
    new_sum_cross_entropy = new_sense_wise_cross_entropy.sum()
    ori_sense_wise_cross_entropy = F.cross_entropy(ori_logits, label_tensor, weight=torch.tensor([1, 8], dtype=torch.float, device=device), reduction='none')
    ori_sum_cross_entropy = ori_sense_wise_cross_entropy.sum()

    instance_wise_sum_cross_entropy = []
    for i in range(num_instances):
        instance_wise_sum_cross_entropy.append(ori_sense_wise_cross_entropy[sep_pos[i]:sep_pos[i+1]].sum())

    WSD_reward = sample_probs[selected_instance_idx]*(new_sum_cross_entropy-ori_sum_cross_entropy)/num_instances
    instance_select_reward = torch.matmul(sample_probs.unsqueeze(0), torch.stack(instance_wise_sum_cross_entropy).unsqueeze(1))/num_instances

    reward = alpha*WSD_reward + (1-alpha)*instance_select_reward

    return reward, WSD_reward, instance_select_reward


def compute_reward2(new_logits, ori_logits, label_tensor, sep_pos, num_instances):
    new_sense_wise_cross_entropy = F.cross_entropy(new_logits, label_tensor, reduction='none')
    new_sum_cross_entropy = new_sense_wise_cross_entropy.sum()
    ori_sense_wise_cross_entropy = F.cross_entropy(ori_logits, label_tensor, reduction='none')
    ori_sum_cross_entropy = ori_sense_wise_cross_entropy.sum()

    instance_wise_sum_cross_entropy = []
    for i in range(num_instances):
        instance_wise_sum_cross_entropy.append(ori_sense_wise_cross_entropy[sep_pos[i]:sep_pos[i + 1]].sum())

    WSD_reward = (new_sum_cross_entropy - ori_sum_cross_entropy) / num_instances

    return WSD_reward


def parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--mode",
                        default='RL-train',
                        type=str,
                        choices=["bert-pretrain", "eval-baseline", "RL-train", 'RL-eval'],
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
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=150,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=80,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--neg_pos_ratio",
                        default=None,
                        type=float,
                        help="Ratio of negative training examples over positive ones.")
    parser.add_argument("--eval_batch_size",
                        default=10,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
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
    parser.add_argument('--num_sample',
                        type=int,
                        default=5,
                        help="Number of sampling during agent training")
    parser.add_argument("--checkpoint",
                        default='output/3_pytorch_model.bin',
                        type=str,
                        help="The saved checkpoint model path to load.")
    parser.add_argument("--eval_dataset",
                        default='dev',
                        type=str,
                        help="The saved checkpoint model path to load.")
    args = parser.parse_args()

    return args


def bert_pretrain(model, dataset):
    model.train()
    #sampler = SequentialSampler(dataset)
    #sampler = RandomSampler(dataset)
    sampler = NegDownSampler(dataset, neg_pos_ratio=args.neg_pos_ratio)
    bert_pretrain_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
                                          collate_fn=lambda x: zip(*x))
    num_train_optimization_steps = int(
        len(sampler) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    observe_interval = 500
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    # Assign loss function
    # loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([10/17, 80/17], dtype=torch.float, device=device), ignore_index=-1)
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 8], dtype=torch.float, device=device), ignore_index=-1)
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
        sampler.refresh_sampler()
        wf = open(os.path.join(args.output_dir, 'log_%d.txt' % epoch), 'w')
        tr_loss = 0
        for step, batch in enumerate(tqdm(bert_pretrain_dataloader, desc="Iteration"), start=1):
            guid, cand_sense_key, input_ids, input_mask, segment_ids, start_id, end_id, label = batch

            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            input_mask_tensor = torch.tensor(input_mask, dtype=torch.long, device=device)
            segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long, device=device)

            label_tensor = torch.tensor(label, dtype=torch.long, device=device)

            batch_size, seq_len = input_ids_tensor.size()
            # The mask has 1 for real target
            selection_mask_tensor = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                selection_mask_tensor[i][start_id[i]:end_id[i]] = 1

            logits = model(input_ids_tensor, input_mask_tensor, segment_ids_tensor, selection_mask_tensor)
            loss = loss_function(logits, label_tensor)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item() * args.gradient_accumulation_steps

            if (step % args.gradient_accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step % observe_interval == 0:
                print('Epoch: %d, Step: %d, avg_loss: %.4f' % (epoch, step, (tr_loss/observe_interval)))
                wf.write('Epoch: %d, Step: %d, avg_loss: %.4f\n' % (epoch, step, (tr_loss / observe_interval)))
                tr_loss = 0
        wf.close()
        # Save a trained model
        logger.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, '_'.join((str(epoch), WEIGHTS_NAME)))
        torch.save(model_to_save.state_dict(), output_model_file)


def eval_baseline(model, dataset):
    model.eval()

    sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: zip(*x))
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

    wf = open(os.path.join(args.output_dir, 'eval_log.txt'), 'w')
    tr_loss = 0
    total_num_correct_pred = 0
    total_num_example = 0
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration"), start=1):
        guid, cand_sense_key, input_ids, input_mask, segment_ids, start_id, end_id, label = batch

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long, device=device)
        segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long, device=device)
        label_tensor = torch.tensor(label, dtype=torch.long, device=device)

        batch_size, seq_len = input_ids_tensor.size()
        # The mask has 1 for real target
        selection_mask_tensor = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
        for i in range(batch_size):
            selection_mask_tensor[i][start_id[i]:end_id[i]] = 1

        with torch.no_grad():
            logits = model(input_ids_tensor, input_mask_tensor, segment_ids_tensor, selection_mask_tensor)
        loss = loss_function(logits, label_tensor)
        tr_loss += loss.item()

        pred_tensor = logits.argmax(dim=1)
        total_num_correct_pred += (pred_tensor==label_tensor).sum().item()
        total_num_example += batch_size

        pred_list = pred_tensor.tolist()
        probs = F.softmax(logits, dim=-1)
        result_batch = [(pred_list[i], probs[i][0].item(), probs[i][1].item()) for i in range(batch_size)]
        for result in result_batch:
            wf.write(str(result[0])+' '+str(result[1])+' '+str(result[2])+'\n')
    wf.close()
    print('Avg_loss: %.4f' % (tr_loss / len(dataset)))
    print('Label accuracy: %.2f' % (total_num_correct_pred/total_num_example))


def agent_pretrain(base_model, agent_model, dataset):
    base_model.eval()
    agent_model.train()
    #sampler = SequentialSampler(dataset)
    sampler = RandomSampler(dataset)
    agent_pretrain_dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=lambda x: x)
    agent_optimizer = AdamW(agent_model.parameters(), lr=args.learning_rate,weight_decay=5)
    observe_interval = 100

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        wf = open(os.path.join(args.output_dir, 'log_agent_train_%d.txt' % epoch), 'w')
        accum_batch_size = 0
        total_reward = 0
        total_WSD_reward = 0
        total_instance_select_reward = 0
        for step, batch in enumerate(tqdm(agent_pretrain_dataloader, desc="Iteration"), start=1):
            cur_sentence = batch[0]
            #cur_sentence = dataset[8]
            valid_instances, sep_pos, feature_batch = convert_sentence_to_feature_batch(cur_sentence, args.max_seq_length, tokenizer)
            guid, cand_sense_key, input_ids, input_mask, segment_ids, start_id, end_id, label = zip(*feature_batch)

            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            input_mask_tensor = torch.tensor(input_mask, dtype=torch.long, device=device)
            segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long, device=device)

            label_tensor = torch.tensor(label, dtype=torch.long, device=device)

            batch_size, seq_len = input_ids_tensor.size()
            accum_batch_size += batch_size
            # The mask has 1 for real target
            selection_mask_tensor = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                selection_mask_tensor[i][start_id[i]:end_id[i]] = 1
            with torch.no_grad():
                ori_logits, final_target_hidden_batch_tensor, mention_aware_gloss_hidden_tensors_list = \
                    base_model(input_ids_tensor, input_mask_tensor, segment_ids_tensor, selection_mask_tensor, output_target_hiddens=True)
            ori_probs = F.softmax(ori_logits, dim=-1)
            pred_list = []
            for i in range(len(valid_instances)):
                pred = ori_probs[sep_pos[i]: sep_pos[i + 1], 1].argmax().item()
                #pred = valid_instances[i].get_gold_label_index()
                pred_list.append(pred)

            tmp_reward = 0
            tmp_WSD_reward = 0
            tmp_instance_select_reward= 0

            result = agent_model(base_model, valid_instances, final_target_hidden_batch_tensor,
                                                                                     mention_aware_gloss_hidden_tensors_list, sep_pos, pred_list,
                                                                                     mode='single-step-train', num_sample=args.num_sample)
            new_logits_list, next_sample_probs, selected_instance_idx_list = result
            for new_logits, selected_instance_idx in zip(new_logits_list, selected_instance_idx_list):
                new_probs = F.softmax(new_logits, dim=-1)
                new_pred_list = []
                for i in range(len(valid_instances)):
                    pred = new_probs[sep_pos[i]: sep_pos[i + 1], 1].argmax().item()
                    new_pred_list.append(pred)
                reward, WSD_reward, instance_select_reward = compute_reward(new_logits, ori_logits, label_tensor, sep_pos,
                                                                            len(valid_instances), next_sample_probs,
                                                                            selected_instance_idx, alpha=0.5)
                tmp_reward += reward
                tmp_WSD_reward += WSD_reward.item()
                tmp_instance_select_reward += instance_select_reward.item()
            tmp_reward.backward()
            if accum_batch_size >= args.train_batch_size:
                clip_grad_norm_(agent_model.parameters(), 1.0, norm_type=2)
                agent_optimizer.step()
                agent_optimizer.zero_grad()
                accum_batch_size = 0

            total_reward += tmp_reward.item() / args.num_sample
            total_WSD_reward += tmp_WSD_reward / args.num_sample
            total_instance_select_reward += tmp_instance_select_reward / args.num_sample
            if step % observe_interval == 0:
                print('Epoch: %d, Step: %d, avg_reward: %.4f, avg_WSD_reward: %.4f, avg_IS_reward: %.4f' %
                      (epoch, step, (total_reward / observe_interval), (total_WSD_reward / observe_interval), (total_instance_select_reward / observe_interval)))
                wf.write('Epoch: %d, Step: %d, avg_reward: %.4f, avg_WSD_reward: %.4f, avg_IS_reward: %.4f\n' %
                         (epoch, step, (total_reward / observe_interval), (total_WSD_reward / observe_interval), (total_instance_select_reward / observe_interval)))
                total_reward = 0
                total_WSD_reward = 0
                total_instance_select_reward = 0
        wf.close()
        # Save a trained model
        logger.info("** ** * Saving agent model ** ** * ")
        model_to_save = agent_model.module if hasattr(agent_model, 'module') else agent_model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, '_'.join((str(epoch), 'agent.bin')))
        torch.save(model_to_save.state_dict(), output_model_file)


def eval(base_model, agent_model, dataset):
    base_model.eval()
    agent_model.eval()
    sampler = SequentialSampler(dataset)
    # sampler = RandomSampler(dataset)
    agent_pretrain_dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=lambda x: x)
    agent_optimizer = AdamW(agent_model.parameters(), lr=args.learning_rate, weight_decay=1)
    observe_interval = 100

    wf = open(os.path.join(args.output_dir, 'RL_eval_log.txt'), 'w')
    total_num_correct_pred = 0
    total_num_example = 0
    total_WSD_reward = 0
    for step, batch in enumerate(tqdm(agent_pretrain_dataloader, desc="Iteration"), start=1):
        cur_sentence = batch[0]
        #cur_sentence = dataset[1]
        valid_instances, sep_pos, feature_batch = convert_sentence_to_feature_batch(cur_sentence,
                                                                                    args.max_seq_length, tokenizer)
        guid, cand_sense_key, input_ids, input_mask, segment_ids, start_id, end_id, label = zip(*feature_batch)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long, device=device)
        segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long, device=device)

        label_tensor = torch.tensor(label, dtype=torch.long, device=device)

        batch_size, seq_len = input_ids_tensor.size()
        # The mask has 1 for real target
        selection_mask_tensor = torch.zeros(batch_size, seq_len, device=device)
        for i in range(batch_size):
            selection_mask_tensor[i][start_id[i]:end_id[i]] = 1
        with torch.no_grad():
            ori_logits, final_target_hidden_batch_tensor, mention_aware_gloss_hidden_tensors_list = \
                base_model(input_ids_tensor, input_mask_tensor, segment_ids_tensor, selection_mask_tensor,
                           output_target_hiddens=True)
        ori_probs = F.softmax(ori_logits, dim=-1)
        pred_list = []
        for i in range(len(valid_instances)):
            pred = ori_probs[sep_pos[i]: sep_pos[i + 1], 1].argmax().item()
            pred_list.append(pred)
        with torch.no_grad():
            new_logits = agent_model(base_model, valid_instances, final_target_hidden_batch_tensor,
                                     mention_aware_gloss_hidden_tensors_list, sep_pos, pred_list, mode='eval')
        pred_tensor = new_logits.argmax(dim=1)
        total_num_correct_pred += (pred_tensor == label_tensor).sum().item()
        total_num_example += batch_size

        new_pred_list = pred_tensor.tolist()
        new_probs = F.softmax(new_logits, dim=-1)
        result_batch = [(new_pred_list[i], new_probs[i][0].item(), new_probs[i][1].item()) for i in range(batch_size)]
        for result in result_batch:
            wf.write(str(result[0]) + ' ' + str(result[1]) + ' ' + str(result[2]) + '\n')

        WSD_reward = compute_reward2(new_logits, ori_logits, label_tensor, sep_pos, len(valid_instances))

        #wf.close()
        total_WSD_reward += WSD_reward.item() / args.num_sample

        if step % observe_interval == 0:
            print('Step: %d, avg_WSD_reward: %.4f' % (step, total_WSD_reward/observe_interval))
            total_WSD_reward = 0
    wf.close()


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
    if args.mode == 'bert-pretrain':
        print('Loading glossbert dataset from csv file...')
        glossbert_dataset = GlossBERTDataset_for_CGPair_Feature.from_data_csv(
            csv_paths['train'], tokenizer, max_seq_length=args.max_seq_length)
            # 'Evaluation_Datasets/semeval2007/semeval2007_test_token_cls.csv', tokenizer, max_seq_length=args.max_seq_length)
            # 'Training_Corpora/SemCor/semcor_train_token_cls.csv', tokenizer, max_seq_length=args.max_seq_length)
        print('Dumping glossbert dataset ...')
        with open('Training_Corpora/SemCor/train_glossbert_dataset.pkl', 'wb') as wbf:
            pickle.dump(glossbert_dataset, wbf)
        #print('Loading glossbert dataset from pkl file...')
        #with open('Training_Corpora/SemCor/train_glossbert_dataset.pkl', 'rb') as rbf:
            #glossbert_dataset = pickle.load(rbf)
        print("  Num positive examples = %d", len(glossbert_dataset.pos_indexes))
        print("  Num negative examples = %d", len(glossbert_dataset.neg_indexes))
        print("  Num invalid examples = %d", len(glossbert_dataset.invalid_indexes))
        # Load open-source bert
        bert_model = BertModel.from_pretrained(bert_dir)
        model = BaseModel(bert_model).to(device)
        bert_pretrain(model, glossbert_dataset)
    elif args.mode == 'eval-baseline':
        glossbert_dataset = GlossBERTDataset_for_CGPair_Feature.from_data_csv(
            csv_paths[args.eval_dataset], tokenizer, max_seq_length=args.max_seq_length)
        #with open('Evaluation_Datasets/semeval2007/semval2007_glossbert_dataset.pkl', 'rb') as rbf:
            #glossbert_dataset = pickle.load(rbf)
        print("  Num positive examples = %d", len(glossbert_dataset.pos_indexes))
        print("  Num negative examples = %d", len(glossbert_dataset.neg_indexes))
        print("  Num invalid examples = %d", len(glossbert_dataset.invalid_indexes))
        # Load open-source bert
        bert_model = BertModel.from_pretrained(bert_dir)
        model = BaseModel(bert_model).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        eval_baseline(model, glossbert_dataset)
    elif args.mode == 'RL-train':
        glossbert_dataset = GlossBERTDataset_for_Sentence.from_data_csv(
            csv_paths[args.eval_dataset], tokenizer, max_seq_length=args.max_seq_length)
        #with open('Evaluation_Datasets/semeval2007/semval2007_glossbert_dataset.pkl', 'rb') as rbf:
            #glossbert_dataset = pickle.load(rbf)
        # Load open-source bert
        bert_model = BertModel.from_pretrained(bert_dir)
        base_model = BaseModel(bert_model).to(device)
        base_model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        agent_model = Agent(hidden_size=768, init_classifier=base_model.classifier).to(device)
        #agent_model.load_state_dict(torch.load('output/0_agent.bin', map_location=torch.device('cpu')))
        agent_pretrain(base_model, agent_model, glossbert_dataset)
    elif args.mode == 'RL-eval':
        glossbert_dataset = GlossBERTDataset_for_Sentence.from_data_csv(
            csv_paths[args.eval_dataset], tokenizer, max_seq_length=args.max_seq_length)
        #with open('Evaluation_Datasets/semeval2007/semval2007_glossbert_dataset.pkl', 'rb') as rbf:
            #glossbert_dataset = pickle.load(rbf)
        # Load open-source bert
        bert_model = BertModel.from_pretrained(bert_dir)
        base_model = BaseModel(bert_model).to(device)
        base_model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        agent_model = Agent(hidden_size=768, init_classifier=base_model.classifier).to(device)
        agent_model.load_state_dict(torch.load('output/0_agent.bin', map_location=torch.device('cpu')))
        eval(base_model, agent_model, glossbert_dataset)