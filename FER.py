import pdb
import re
import sys
from transformers import AdamW, AutoTokenizer
from transformers import  BertModel, BertForSequenceClassification
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
import os
import copy
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, Subset
from torch.utils.data import DataLoader, RandomSampler
import argparse
from prettytable import PrettyTable
from fever_score import fever_score

def init(seed):
    init_seed = seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
init(42)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def pre_processing_evidence_extractor(sentence_train, gold_evidence, labels, sentence_gold, ids):
    input_ids = []
    attention_masks = []
    gold_evidence_mask = []
    input_ids_gold = []
    attention_masks_gold = []
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
        encoded_dict_gold = tokenizer.encode_plus(
            sentence_gold[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        input_ids_gold.append(encoded_dict_gold['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks_gold.append(encoded_dict_gold['attention_mask'])
        # remove everyting in [CLS] ... [SEP] pair with regex, shortest match
        gold_evidence_only = gold_evidence[i].split(' [SEP] ')[:-1]
        sentence_evidence_only = re.sub(r' \[CLS\].*?\[SEP\] ', '', sentence_train[i]).split(' [SEP] ')[:-1]
        # remove space
        gold_evidence_only = [s.replace(' ', '') for s in gold_evidence_only]
        sentence_evidence_only = [s.replace(' ', '') for s in sentence_evidence_only]
        is_gold_evidence = [1 if sentence_evidence_only[i] in gold_evidence_only else 0 for i in
                            range(len(sentence_evidence_only))]
        is_gold_evidence += [0] * (25 - len(is_gold_evidence))
        gold_evidence_mask.append(is_gold_evidence)
    ids = torch.tensor(ids)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    input_ids_gold = torch.cat(input_ids_gold, dim=0)
    attention_masks_gold = torch.cat(attention_masks_gold, dim=0)
    gold_evidence_mask = torch.tensor(gold_evidence_mask)
    labels = torch.tensor(labels)
    train_dataset = TensorDataset(input_ids, attention_masks, gold_evidence_mask, labels, input_ids_gold,
                                  attention_masks_gold, ids)
    return train_dataset, tokenizer

class bert_claim(nn.Module):
    def __init__(self):
        super(bert_claim, self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-uncased' )
        self.dense = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.6)
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:,0,:]
        state = self.dropout(bert_cls_hidden_state)
        linear_output = self.dense(state)
        return linear_output


class EvidenceExtractor(nn.Module):
    def __init__(self):
        super(EvidenceExtractor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


def prepareToTrain(sentence, gold_evidence, labels, sentence_gold, ids):
    dataset, tokenizer = pre_processing_evidence_extractor(sentence, gold_evidence, labels, sentence_gold, ids)
    val_dataset = Subset(dataset, [i for i in range(19980)])
    train_dataset = Subset(dataset, [i for i in range(19980, len(dataset))])


    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=RandomSampler(val_dataset),
        batch_size=128
    )
    model = EvidenceExtractor()
    bert_model = bert_claim()
    return bert_model, model, train_dataloader, val_dataloader


def format_evidence(evidence):
    full_text_list = [item['text'] for item in evidence.values()]
    full_text_list = [item for item in full_text_list][:25]
    return ' [SEP] '.join(full_text_list) + ' [SEP] '


def get_claim_evidence_sentence():
    datalist = json.load(open('./data/dev_dict.json', 'r', encoding='utf-8')) \
               + json.load(open('./data/train_dict.json', 'r', encoding='utf-8'))
    claims = [' [CLS] ' + row['claim'] + ' [SEP] ' for row in datalist]
    evidences = [format_evidence(row['evidence']) for row in datalist]
    ids = [int(row["id"]) for row in datalist]
    gold_evidences = []
    for row in datalist:
        if row['label'] == 2:
            gold_evidences.append(format_evidence(row['evidence']))
        else:
            gold_evidences.append(format_evidence(row['golden evidence']))

    sentence = [claim + evidence for claim, evidence in zip(claims, evidences)]
    sentence_gold = [claim + evidence for claim, evidence in zip(claims, gold_evidences)]
    data_dict = {}
    with open("./data/train.json") as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = data["evidence"][:25]
    with open("./data/dev.json") as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = data["evidence"][:25]
    return sentence, sentence_gold, ids,  data_dict


def get_gold_evidence_sentence():
    datalist = json.load(open('./data/dev_dict.json', 'r', encoding='utf-8')) \
               + json.load(open('./data/train_dict.json', 'r', encoding='utf-8'))
    gold_evidences = []
    for row in datalist:
        if row['label'] == 2:
            gold_evidences.append('[SEP]')
        else:
            gold_evidences.append(format_evidence(row['golden evidence']))
    return gold_evidences


def eval_model_retrieval(device, bert_model, extractor_model, extractor_bert_model,  val_dataloader, Data_dict_id, retrieval_output = "output/retrieval_evidence_feedback.json",calim_output = "output/claim_veracity_feedback.json", gold_file="data/golden_dev.json"):
    data_dict = dict()
    bert_model.eval()
    extractor_model.eval()
    extractor_bert_model.eval()
    with open(gold_file) as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = {"id": data["id"], "evidence": [], "claim": data["claim"]}
            if "label" in data:
                data_dict[data["id"]]["label"] = data["label"]
    evidence_num = 0
    num = 0
    with open(calim_output, "w") as f:
        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[3].to(device)
            ids = batch[6].tolist()
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
                    # sentences = sentence.split('[SEP]')[:-1]
                    pred = outputs[i]
                    evidence = []
                    for j in range(len(sentences)):
                        if pred[j] > 0:
                            evidence.append(sentences[j][:2])
                    data_dict[id]["evidence"] = evidence
                    evidence_num += len(evidence)
                    num += 1

                b_input_mask_claim, b_input_mask_25_evidence = get_input_masks_for_claim_and_evidence(
                    b_input_mask, b_input_ids, device)
                outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
                outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
                b_input_mask_use_extractor = torch.bmm(outputs.unsqueeze(1),
                                                       b_input_mask_25_evidence.float()).squeeze(
                    1) + b_input_mask_claim

                outputs_use_extractor = bert_model(b_input_ids, b_input_mask_use_extractor)
                logits = outputs_use_extractor
            pred_flat = torch.max(logits, 1)[1]
            label_map = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
            for step in range(len(ids)):
                instance = {"id": ids[step], "predicted_label": label_map[pred_flat[step].item()]}
                f.write(json.dumps(instance) + "\n")
    print('-------------------------')
    print(evidence_num/num)
    with open(retrieval_output, "w") as out:
        for data in data_dict.values():
            out.write(json.dumps(data) + "\n")
    predicted_labels = []
    predicted_evidence = []
    actual = []
    ids = dict()
    with open(calim_output, "r") as predictions_file:
        for line in predictions_file:
            ids[json.loads(line)["id"]] = len(predicted_labels)  # id映射0-n-1
            predicted_labels.append(json.loads(line)["predicted_label"])
            predicted_evidence.append(0)
            actual.append(0)
    with open("./data/all_dev.json", "r") as allfile:
        for line in allfile:
            if json.loads(line)["id"] not in ids:
                ids[json.loads(line)["id"]] = len(predicted_labels)  # id映射0-n-1
                predicted_labels.append("NOT ENOUGH INFO")
                predicted_evidence.append(0)
                actual.append(0)
    with open(retrieval_output, "r") as predictions_file:
        for line in predictions_file:
            predicted_evidence[ids[json.loads(line)["id"]]] = json.loads(line)["evidence"]

    with open("./data/dev_eval.json", "r") as actual_file:
        for line in actual_file:
            actual[ids[json.loads(line)["id"]]] = json.loads(line)
    predictions = []
    for ev, label in zip(predicted_evidence, predicted_labels):
        predictions.append({"predicted_evidence": ev, "predicted_label": label})
    score, acc, precision, recall, f1 = fever_score(predictions, actual)

    tab = PrettyTable()
    tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
    tab.add_row((round(score, 4), round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)))
    print(tab)
    return round(score, 4), acc, f1, tab
# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Bert Classification For CHEF')
parser.add_argument('--noload', action='store_false', help='if present, do not load any saved model')
parser.add_argument('--cuda', type=str, default="1", help='appoint GPU devices')
parser.add_argument('--num_labels', type=int, default=3, help='num labels of the dataset')
parser.add_argument('--max_length', type=int, default=512, help='max token length of the sentence for bert tokenizer')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--initial_lr', type=float, default=2e-5, help='initial learning rate')
parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
parser.add_argument('--epochs', type=int, default=4, help='training epochs for labeled data')
parser.add_argument('--total_epochs', type=int, default=10, help='total epochs of the RL learning')
parser.add_argument("--iteration_dis_step", default=400, type=int)
parser.add_argument("--iteration_step", default=200, type=int)
parser.add_argument("--accumulation", default=1, type=int)
# parser.add_argument("--local_rank" , default=os.getenv('LOCAL_RANK', -1), type=int)
args = parser.parse_args()

def main(argv=None):
    init(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda")
    datalist = json.load(open('./data/dev_dict.json', 'r', encoding='utf-8')) \
               + json.load(open('./data/train_dict.json', 'r', encoding='utf-8'))

    labels = [row['label'] for row in datalist]
    print('====================Init model and dataset...=================')
    sentence, sentence_gold, ids, data_dict = get_claim_evidence_sentence()
    gold_evidence = get_gold_evidence_sentence()
    bert_model, extractor_model, train_dataloader, val_dataloader = \
        prepareToTrain(
            sentence, gold_evidence, labels, sentence_gold, ids
        )

    bert_model.load_state_dict(torch.load('save_model/verifier.bin'))
    bert_model = bert_model.to(device)
    extractor_model = extractor_model.to(device)
    score, Acc, f1, tab_best = \
        train_and_save_extractor_model(extractor_model, bert_model, train_dataloader, val_dataloader,device, data_dict,
                                       no_load=args.noload)
    print(tab_best)

def train_and_save_extractor_model(extractor_model, bert_model, train_dataloader, val_dataloader,  device, Data_dict_id, bert_type='bert',
                                   model_save_dir='model_save', no_load=False):
    best_score, best_Acc, best_f1, tab_best = 0, 0, 0, []
    bert_finetuned = copy.deepcopy(bert_model)
    for p in bert_finetuned.parameters():
                p.requires_grad = False
    def one_hot(x, class_count):
        return torch.eye(class_count)[x, :]
    bert_model = BertForSequenceClassification.from_pretrained(
        './bert-base-uncased',  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=3,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=True,  # Whether the model returns all hidden-states.
    ).to(device)

    weights = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    extractor_epochs = 5
    extractor_model.load_state_dict(torch.load('./save_model/evidenceextractor_cla.bin'))
    bert_model.load_state_dict(torch.load('./save_model/bertforseqcls_for_ext_cla.bin'))
    optimizer = AdamW(
        list(extractor_model.parameters()) + list(bert_model.parameters()),
        lr=1e-6,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps,  # args.adam_epsilon  - default is 1e-8.,
        weight_decay=0.01
    )
    # total_steps = len(train_dataloader) * extractor_epochs * 2 * 2
    total_steps = len(train_dataloader) * extractor_epochs * 2 * 2
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    bestacc = 0

    gloab_step = 0
    rankloss = nn.MarginRankingLoss()
    extractor_model.zero_grad()
    bert_model.zero_grad()
    for epoch_i in range(0, extractor_epochs):
        # ========================================
        #               Training
        # ========================================
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, extractor_epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        extractor_model.train()
        bert_model.train()
        bert_finetuned.eval()
        bert_finetuned.zero_grad()
        for step, batch in enumerate(train_dataloader):
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed), end=' ')
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_input_ids_gold = batch[4].to(device)
            b_input_mask_gold = batch[5].to(device)
            b_labels_01 = batch[2].to(device)
            b_labels = batch[3].to(device)
            extractor_model.train()
            bert_model.train()


            extractor_model.zero_grad()
            bert_model.zero_grad()
            outputs_bert = bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                output_hidden_states=True
            )
            cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
            b_input_mask_claim, b_input_mask_25_evidence = get_input_masks_for_claim_and_evidence(b_input_mask,
                                                                                                  b_input_ids, device)
            outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
            outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
            b_input_mask_use_extractor = torch.bmm(outputs.unsqueeze(1), b_input_mask_25_evidence.float()).squeeze(
                1) + b_input_mask_claim
            y_one_hot = one_hot(batch[3].cpu(), 3)
            y_one_hot = y_one_hot.to(device)
            outputs_use_extractor = bert_finetuned(b_input_ids, b_input_mask_use_extractor)
            logits = F.softmax(outputs_use_extractor, dim=1)
            loss_feedback_retrieval = torch.sum(torch.mul(logits, y_one_hot), dim=1)
            outputs_use_extractor_gold = bert_finetuned(b_input_ids_gold, b_input_mask_gold)
            logits = F.softmax(outputs_use_extractor_gold, dim=1)
            loss_feedback_gold = torch.sum(torch.mul(logits, y_one_hot), dim=1)
            target = torch.ones_like(loss_feedback_gold)
            rk_loss = rankloss(loss_feedback_retrieval, loss_feedback_gold, target)/100
            print(rk_loss.item())
            rk_loss.requires_grad_(True)
            rk_loss.backward()

            nn.utils.clip_grad_value_(extractor_model.parameters(), 1)
            nn.utils.clip_grad_value_(bert_model.parameters(), 1)
            if gloab_step % args.accumulation == 0:
                optimizer.step()
                scheduler.step()
                extractor_model.zero_grad()
                bert_model.zero_grad()
                bert_finetuned.zero_grad()

            gloab_step += 1
        # ========================================
        #               Validation
        # ========================================
            if gloab_step%(args.accumulation*400)==0:
                score, Acc, f1, tab = eval_model_retrieval(device, bert_finetuned, extractor_model, bert_model, val_dataloader,
                                                      Data_dict_id)


                if f1 > bestacc:
                    bestacc = f1
                    best_score, best_Acc, best_f1, tab_best = score, Acc, f1, tab
                    torch.save(extractor_model.state_dict(), './save_model/evidenceextractor_fer.bin')
                    torch.save(bert_model.state_dict(), './save_model/bertforseqcls_fer.bin')

    return best_score, best_Acc, best_f1, tab_best




def get_input_masks_for_claim_and_evidence(b_input_mask, b_input_ids, device):
    # will use evidence extractor, and alter b_input_mask
    b_input_mask_claim = torch.zeros_like(b_input_mask)
    for bz in range(b_input_mask.shape[0]):
        for idx in range(b_input_mask.shape[1]):
            b_input_mask_claim[bz][idx] = 1
            if b_input_ids[bz][idx] == 102:
                break
    b_input_mask_25_evidence = torch.zeros([b_input_mask.shape[0], 25, b_input_mask.shape[1]], dtype=torch.long).to(device)
    for bz in range(b_input_mask.shape[0]):
        count = -1
        for idx in range(b_input_mask.shape[1]):
            if count >= 25:
                break
            if count >= 0:
                b_input_mask_25_evidence[bz][count][idx] = 1
            if b_input_ids[bz][idx] == 102:
                count += 1
    return b_input_mask_claim, b_input_mask_25_evidence

if __name__ == '__main__':
    sys.exit(main())
