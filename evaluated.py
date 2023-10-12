import sys
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import os
import json

from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler
import argparse

def pre_processing(sentence_train, bert_type):
    input_ids = []
    attention_masks = []
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    train_dataset = TensorDataset(input_ids, attention_masks)
    return train_dataset, tokenizer


def pre_processing_evidence_extractor(sentence_train, ids):
    input_ids = []
    attention_masks = []
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased')
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    ids = torch.tensor(ids)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    train_dataset = TensorDataset(input_ids, attention_masks, ids)
    return train_dataset, tokenizer


class EvidenceExtractor(nn.Module):
    def __init__(self):
        super(EvidenceExtractor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加Dropout层
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加Dropout层
            nn.Linear(256, 50)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


def prepareToTrain(sentence, ids):
    dataset, tokenizer = pre_processing_evidence_extractor(sentence, ids)
    val_dataloader = DataLoader(
        dataset,
        # sampler=DistributedSampler(val_dataset),
        sampler=RandomSampler(dataset),
        batch_size=128
    )
    model = EvidenceExtractor()
    # model = nn.DataParallel(model)
    bert_model = BertForSequenceClassification.from_pretrained(
        './bert-base-uncased',  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=3,  # The number of output labels--3
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # bert_model = nn.DataParallel(bert_model)
    return bert_model, model, val_dataloader


def format_evidence(evidence):
    full_text_list = [item[2] for item in evidence]
    full_text_list = [item for item in full_text_list if len(item) > 5][:25]
    return ' [SEP] '.join(full_text_list) + ' [SEP] '


def get_claim_evidence_sentence(path, path_all):
    claims = []
    claims_ids = {}
    evidences = []
    ids = []
    with open(path_all) as f:
        for line in f:
            data = json.loads(line)
            claims_ids[data["id"]] = data['claim']

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            evidences.append(format_evidence(data['evidence']))
            claims.append(' [CLS] ' + claims_ids[data["id"]] + ' [SEP] ')
            ids.append(data["id"])
    sentence = [claim + evidence for claim, evidence in zip(claims, evidences)]
    data_dict = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = data["evidence"][:25]
    return sentence, ids,  data_dict

def extract_evidence_val(retrieval_output, Data_dict_id):
    claims = []
    evidences = []
    ids = []
    with open(retrieval_output) as f:
        for line in f:
            data = json.loads(line)
            if data['evidence'] == []:
                if data['id'] not in Data_dict_id:
                    evidences.append(' [SEP] ')
                else:
                    full_text_list = [item[2] for item in Data_dict_id[data['id']]]
                    full_text_list = full_text_list
                    evidences.append(' [SEP] '.join(full_text_list) + ' [SEP] ')
            else:
                full_text_list = [item[2] for item in data['evidence']]
                full_text_list = full_text_list
                evidences.append(' [SEP] '.join(full_text_list) + ' [SEP] ')
            claims.append(' [CLS] ' + data['claim'] + ' [SEP] ')
            ids.append(data["id"])
    sentence_train = [claim + evidence for claim, evidence in zip(claims, evidences)]
    input_ids = []
    attention_masks = []
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased')
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    ids = torch.tensor(ids)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    train_dataset = TensorDataset(input_ids, attention_masks, ids)
    val_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=128
    )
    return val_dataloader
def eval_model_retrieval(device, extractor_model, extractor_bert_model,  val_dataloader, Data_dict_id, retrieval_output="output/retrieval_evidence_test_only_plau.json", gold_file="data/all_test.json"):
    data_dict = dict()
    extractor_model.eval()
    extractor_bert_model.eval()
    with open(gold_file) as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = {"id": data["id"], "evidence": [], "claim": data["claim"], "predicted_label": ' '}
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        ids = batch[2].tolist()
        with torch.no_grad():
            outputs_bert = extractor_bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                output_hidden_states=True
            )
            cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
            outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
            outputs = F.softmax(outputs, dim=-1)[:, :, -1]
            # _, preds = torch.max(outputs, 2)
            # outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
            outputs = outputs.tolist()
            # preds = preds.tolist()
            for i, id in enumerate(ids):
                sentences = Data_dict_id[id]
                pred = outputs[i]
                evidence = []
                for j in range(len(sentences)):
                    evidence.append(sentences[j][:3]+[pred[j]])
                data_dict[id]["evidence"] = evidence
    with open(retrieval_output, "w") as out:
        for data in data_dict.values():
            out.write(json.dumps(data) + "\n")
def eval_model_retrieval_test(device, extractor_model, extractor_bert_model,  val_dataloader, Data_dict_id, retrieval_output="output/retrieval_evidence_test.json", calim_output="output/claim_veracity_test.json",final_output="output/prediction.json", gold_file="data/all_test.json"):
    data_dict = dict()
    extractor_model.eval()
    extractor_bert_model.eval()
    with open(gold_file) as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = {"id": data["id"], "evidence": [], "claim": data["claim"], "predicted_label":' '}
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        ids = batch[2].tolist()
        with torch.no_grad():
            outputs_bert = extractor_bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                output_hidden_states=True
            )
            cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
            outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
            outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
            outputs = outputs.tolist()
            for i, id in enumerate(ids):
                sentences = Data_dict_id[id]
                pred = outputs[i]
                evidence = []
                for j in range(len(sentences)):
                    if pred[j] > 0:
                        evidence.append(sentences[j][:3])
                data_dict[id]["evidence"] = evidence
    with open(retrieval_output, "w") as out:
        for data in data_dict.values():
            out.write(json.dumps(data) + "\n")

def eval_model_retrieval_baseline(device, bert_model, extractor_model, extractor_bert_model,  val_dataloader, Data_dict_id, retrieval_output="output/retrieval_evidence_baseline.json", gold_file="data/all_dev.json"):
    data_dict = dict()
    bert_model.eval()
    extractor_model.eval()
    extractor_bert_model.eval()
    with open(gold_file) as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = {"id": data["id"], "evidence": [], "claim": data["claim"],"label": ' '}
            if 'label' in data:
                data_dict[data["id"]]["label"] = data["label"]
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        ids = batch[2].tolist()
        with torch.no_grad():
            outputs_bert = extractor_bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                output_hidden_states=True
            )
            cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
            outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
            outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
            outputs = outputs.tolist()
            for i, id in enumerate(ids):
                sentences = Data_dict_id[id]
                pred = outputs[i]
                evidence = []
                for j in range(len(sentences)):
                    if pred[j] > 0:
                        evidence.append(sentences[j])
                data_dict[id]["evidence"] = evidence

    with open(retrieval_output, "w") as out:
        for data in data_dict.values():
            out.write(json.dumps(data) + "\n")






# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Bert Classification For CHEF')
parser.add_argument('--noload', action='store_false', help='if present, do not load any saved model')
parser.add_argument('--cuda', type=str, default="1", help='appoint GPU devices')
parser.add_argument('--num_labels', type=int, default=3, help='num labels of the dataset')
parser.add_argument('--max_length', type=int, default=512, help='max token length of the sentence for bert tokenizer')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--initial_lr', type=float, default=2e-5, help='initial learning rate')
parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
parser.add_argument('--epochs', type=int, default=4, help='training epochs for labeled data')
parser.add_argument('--total_epochs', type=int, default=10, help='total epochs of the RL learning')
parser.add_argument("--iteration_dis_step", default=500, type=int)
parser.add_argument("--iteration_step", default=200, type=int)
parser.add_argument("--accumulation", default=5, type=int)
parser.add_argument("--type", default="dev", type=str)
parser.add_argument("--dev", default=1, type=int)
parser.add_argument("--test", default=1, type=int)
# parser.add_argument("--local_rank" , default=os.getenv('LOCAL_RANK', -1), type=int)
args = parser.parse_args()
def init(seed):
    init_seed = seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def main(argv=None):
    init(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda")
    path = "./data/"+args.type+".json"
    print('====================Init model and dataset...=================')
    sentence, ids, data_dict = get_claim_evidence_sentence(path,'./data/all_'+args.type+'.json')
    bert_model, extractor_model,  val_dataloader = \
        prepareToTrain(
            sentence, ids
        )
    extractor_bert_model = BertForSequenceClassification.from_pretrained(
        './bert-base-uncased',  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=3,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=True,  # Whether the model returns all hidden-states.
    ).to(device)
    extractor_model = extractor_model.to(device)
    bert_model = bert_model.to(device)

    extractor_model.load_state_dict(torch.load('./save_model/evidenceextractor_fer.bin'))
    extractor_bert_model.load_state_dict(torch.load('./save_model/bertforseqcls_fer.bin'))
   
    eval_model_retrieval_test(device, extractor_model, extractor_bert_model, val_dataloader,
                              data_dict, retrieval_output="output/dev_test.json",gold_file="data/all_dev.json")
  


if __name__ == '__main__':
    sys.exit(main())