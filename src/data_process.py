import json
import re
path = 'data/all_train.json'
golden = 'data/golden_train.json'
data_total = {}
label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}


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
with open(path, encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        id = data["id"]
        if id not in data_total:
            data_total[data["id"]] = {"id": data["id"], "evidence": {},  "golden evidence": {}, "claim": '', "label": "", 'label_c': ''}
        for num, evidence in enumerate(data["evidence"]):
            data_total[data["id"]]["evidence"][num] = {"text": process_sent(evidence[2]), "title": process_wiki_title(evidence[0])}
data_list = []
with open(golden, encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        label = data["label"]
        id = data["id"]
        if id not in data_total:
            continue
        data_total[id]['label'] = label_map[label]
        data_total[id]['label_c'] = label
        data_total[id]['claim'] = data["claim"]
        for num, evidence in enumerate(data["evidence"]):
            data_total[id]["golden evidence"][num] = {"text": process_sent(evidence[2]), "title": process_wiki_title(evidence[0])}
        # print(data_total[data["id"]])
        data_list.append(data_total[data["id"]])

with open('data/train_dict_ori.json','w') as file:
	file.write(json.dumps(data_list, indent=2))#indent为了缩进

