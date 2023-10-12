import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import BertTokenizer

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    sent_a, title, sent_b = sentence
    tokens_a = tokenizer.tokenize(sent_a)

    tokens_b = None
    tokens_t = None
    if sent_b and title:
        tokens_t = tokenizer.tokenize(title)
        tokens_b = tokenizer.tokenize(sent_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4 - len(tokens_t))
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b and tokens_t:
        tokens = tokens + tokens_t + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + len(tokens_t) + 2)
    #print (tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids





def tok2int_list(src_list, tokenizer, max_seq_length, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    for step, sent in enumerate(src_list):
        input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)
    if max_seq_size != -1:
        inp_padding = inp_padding[:max_seq_size]
        msk_padding = msk_padding[:max_seq_size]
        seg_padding = seg_padding[:max_seq_size]
        inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
        msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
        seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding
def process_sent(sentence):
    sentence = re.sub(" LSB.*?RSB", "", sentence)
    sentence = re.sub("LRB RRB ", "", sentence)
    sentence = re.sub("LRB", " ( ", sentence)
    sentence = re.sub("RRB", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)

    return sentence

def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub("LRB", " ( ", title)
    title = re.sub("RRB", " )", title)
    title = re.sub("COLON", ":", title)
    return title
def get_ver_input(evidences, claims, device,args):
    tokenizer = BertTokenizer.from_pretrained('./bert_base', do_lower_case=False)
    inputs = []
    for i, instance in enumerate(evidences):
        claim = claims[i]
        evi_list = list()
        for evidence in instance:
            evi_list.append([process_sent(claim), process_wiki_title(evidence[0]),
                             process_sent(evidence[2])])

        evi_list = evi_list[:5]
        inputs.append(evi_list)
    inp_padding_inputs, msk_padding_inputs, seg_padding_inputs = [], [], []
    for step in range(len(inputs)):
        inp, msk, seg = tok2int_list(inputs[step], tokenizer, args.max_len, 5)
        inp_padding_inputs += inp
        msk_padding_inputs += msk
        seg_padding_inputs += seg
    inp_tensor_input = Variable(
        torch.LongTensor(inp_padding_inputs)).view(-1, 5, args.max_len).to(device)
    msk_tensor_input = Variable(
        torch.LongTensor(msk_padding_inputs)).view(-1, 5, args.max_len).to(device)
    seg_tensor_input = Variable(
        torch.LongTensor(seg_padding_inputs)).view(-1, 5, args.max_len).to(device)
    return (inp_tensor_input, msk_tensor_input, seg_tensor_input)
def get_ver_inputg(evidences, claims, device,args):
    tokenizer = BertTokenizer.from_pretrained('./bert_base', do_lower_case=False)
    inputs = []
    for i, instance in enumerate(evidences):
        claim = claims[i]
        evi_list = list()
        for evidence in instance.values():
            evi_list.append([process_sent(claim), process_wiki_title(evidence["title"]),
                             process_sent(evidence["text"])])

        evi_list = evi_list[:5]
        inputs.append(evi_list)
    inp_padding_inputs, msk_padding_inputs, seg_padding_inputs = [], [], []
    for step in range(len(inputs)):
        inp, msk, seg = tok2int_list(inputs[step], tokenizer, args.max_len, 5)
        inp_padding_inputs += inp
        msk_padding_inputs += msk
        seg_padding_inputs += seg
    inp_tensor_input = Variable(
        torch.LongTensor(inp_padding_inputs)).view(-1, 5, args.max_len).to(device)
    msk_tensor_input = Variable(
        torch.LongTensor(msk_padding_inputs)).view(-1, 5, args.max_len).to(device)
    seg_tensor_input = Variable(
        torch.LongTensor(seg_padding_inputs)).view(-1, 5, args.max_len).to(device)
    return (inp_tensor_input, msk_tensor_input, seg_tensor_input)
