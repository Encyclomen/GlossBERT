import random

from tqdm import tqdm

from model.definition import InputFeatures


def _truncate_seq_pair(tokens_a, tokens_b, max_length, target_end=float('-inf')):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if total_length <= target_end:
            return -1
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

    return 0


def random_word(tokens, tokenizer, mask_prob=0.15):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_single_example_to_features(ori_tokens, max_seq_length, tokenizer):
  #只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
  if len(ori_tokens) > max_seq_length - 2:
      ori_tokens = ori_tokens[0:(max_seq_length - 2)]
  # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
  # (a) 两个句子:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) 单个句子:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # 这里 "type_ids" 主要用于区分第一个第二个句子。
  # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
  # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in ori_tokens:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
  # 创建mask
  input_mask = [1] * len(input_ids)
  # 对于输入进行补0
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  features = (input_ids, input_mask, segment_ids)

  return features


def convert_pair_example_to_features(ori_tokens_a, ori_tokens_b, max_seq_length, tokenizer, mask_prob=0):
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(ori_tokens_a, ori_tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(ori_tokens_a, tokenizer, mask_prob=mask_prob)
    tokens_b, t2_label = random_word(ori_tokens_b, tokenizer, mask_prob=mask_prob)
    # concatenate lm labels and account for CLS, SEP, SEP
    # lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    # assert len(lm_label_ids) == max_seq_length

    features = (input_ids, input_mask, segment_ids)
    return features


def convert_example_to_features(example, max_seq_length, tokenizer):
    guid = example.guid
    cand_sense_key = example.cand_sense_key
    text_a = example.text_a
    text_b = example.text_b
    label = example.label
    start_id = example.start_id
    end_id = example.end_id
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    input_ids1, input_mask1, segment_ids1 = convert_single_example_to_features(tokens_a, max_seq_length, tokenizer)
    input_ids2, input_mask2, segment_ids2 = convert_single_example_to_features(tokens_b, max_seq_length, tokenizer)

    return guid, cand_sense_key, input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, start_id, end_id, label


def convert_example_to_features2(example, max_seq_length, tokenizer, mask_prob=0):
    guid = example.guid
    cand_sense_key = example.cand_sense_key
    text_a = example.text_a
    text_b = example.text_b
    label = example.label
    start_id = example.start_id
    end_id = example.end_id
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    input_ids, input_mask, segment_ids = convert_pair_example_to_features(tokens_a, tokens_b, max_seq_length, tokenizer, mask_prob=mask_prob)

    return guid, cand_sense_key, input_ids, input_mask, segment_ids, start_id, end_id, label


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode='classification'):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens = example.text_a.split(' ')
        target_start = example.start_id
        target_end = example.end_id
        bert_tokens = []

        bert_tokens.append("[CLS]")
        for length in range(len(orig_tokens)):
            if length == target_start:
                target_to_tok_map_start = len(bert_tokens)
            if length == target_end:
                target_to_tok_map_end = len(bert_tokens)
                break
            bert_tokens.extend(tokenizer.tokenize(orig_tokens[length]))
        # bert_tokens.append("[SEP]")
        bert_tokens = tokenizer.tokenize(example.text_a)
        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        segment_ids = [0] * len(bert_tokens)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(bert_tokens, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # assert len(bert_tokens) <= max_seq_length, "sentence must be shorter than max_seq_length"

        bert_tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # The mask has 1 for real target
        target_mask = [0] * max_seq_length
        for i in range(target_to_tok_map_start, target_to_tok_map_end):
            target_mask[i] = 1

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          target_mask=target_mask))
    return features
