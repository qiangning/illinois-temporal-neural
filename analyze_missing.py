from TemporalDataSet import *
trainset = temprel_set('data/Output4LSTM_Baseline/trainset-temprel.xml')
import json
pair_map = json.loads(open('/shared/preprocessed/sssubra2/embeddings/timeline_bigram_counts.json').readlines()[0])
for k in pair_map: 
    for k2 in pair_map[k]: 
        if 'before' not in pair_map[k][k2]: 
            pair_map[k][k2]['before'] = 0 
        if 'after' not in pair_map[k][k2]: 
            pair_map[k][k2]['after'] = 0 
missing = set()
for i, rel in enumerate(trainset.temprel_ee):
    if rel.lemma[rel.event_ix[0]] in pair_map and rel.lemma[rel.event_ix[1]] in pair_map[rel.lemma[rel.event_ix[0]]]:
        if pair_map[rel.lemma[rel.event_ix[0]]][rel.lemma[rel.event_ix[1]]]['before']+pair_map[rel.lemma[rel.event_ix[0]]][rel.lemma[rel.event_ix[1]]]['after'] < 10:
            missing.add((i))
    elif rel.lemma[rel.event_ix[1]] in pair_map and rel.lemma[rel.event_ix[0]] in pair_map[rel.lemma[rel.event_ix[1]]]:
        if pair_map[rel.lemma[rel.event_ix[1]]][rel.lemma[rel.event_ix[0]]]['before']+pair_map[rel.lemma[rel.event_ix[1]]][rel.lemma[rel.event_ix[0]]]['after'] < 10:
            missing.add((i))
    else:
        missing.add((i))
print(len(missing))
from pairwise_ffnn_pytorch import *
import torch
model = VerbNet(32837, 0.3, 200, 1)
checkpoint = torch.load('/shared/preprocessed/sssubra2/embeddings/models/Timelines/pairwise_model_0.3_200_1.pt')
model.load_state_dict(checkpoint['model_state_dict'])
verb_i_map = {}
with open('/shared/preprocessed/sssubra2/embeddings/models/Timelines/embeddings_0.3_200_1_timelines.txt') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        verb_i_map[line.split()[0]] = i
for i in list(missing):
    if trainset.temprel_ee[i].label not in {'AFTER', 'BEFORE'}:
        missing.remove(i)
count = 0
import numpy as np
for i in missing:
    if trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[0]] not in verb_i_map or trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[1]] not in verb_i_map:
        continue
    value = model(torch.from_numpy(np.array([[verb_i_map[trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[0]]], verb_i_map[trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[1]]]]])))[0][0]
    if value > 0.5 and trainset.temprel_ee[i].label == 'BEFORE':
        count += 1
    if value < 0.5 and trainset.temprel_ee[i].label == 'AFTER':
        count += 1
print('Correct prediction: ' + str(count/float(len(missing))))
count = 0
for i in missing:
    if trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[0]] not in verb_i_map or trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[1]] not in verb_i_map:
        continue
    value = model(torch.from_numpy(np.array([[verb_i_map[trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[0]]], verb_i_map[trainset.temprel_ee[i].lemma[trainset.temprel_ee[i].event_ix[1]]]]])))[0][0]
    if value < 0.5 and trainset.temprel_ee[i].label == 'BEFORE':
        count += 1
    if value > 0.5 and trainset.temprel_ee[i].label == 'AFTER':
        count += 1
print('Incorrect prediction: ' + str(count/float(len(missing))))
