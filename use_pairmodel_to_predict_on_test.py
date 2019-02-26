import numpy as np
import pickle as pkl
from TemporalDataSet import temprel_set
from exp_myLSTM import bigramGetter_fromNN, confusion2prf
from extract_bigram_stats import *

ratio = 0.3
emb_size = 200
layer = 1
splitter = " "
timeline_kb = True
print("---------")
print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
if timeline_kb:
    emb_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
    mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
else:
    emb_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_%.1f_%d_%d_temprob.txt' % (ratio, emb_size, layer)
    mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/pairwise_model_%.1f_%d_%d_TemProb.pt' % (ratio, emb_size, layer)
# bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size,splitter = splitter)

# TemProb raw stats
# bigramGetter = pkl.load(open("/shared/preprocessed/qning2/temporal/TemProb/temporal_bigram_stats.pkl", 'rb'))

# Timeline raw stats
bigramGetter=pkl.load(open("/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats_new_all.pkl",'rb'))
testset = temprel_set("data/Output4LSTM_Baseline/testset-temprel.xml")
confusion = np.zeros((4, 4), dtype=int)
label2ix = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}

for i, temprel in enumerate(testset.temprel_ee):
    stats = bigramGetter.getBigramStatsFromTemprel(temprel)
    if stats[0][0] >= stats[0][1]:
        pred = 0
    else:
        pred = 1
    gold = label2ix[temprel.label]
    confusion[gold][pred] += 1
print(confusion)
print("Acc")
print(1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion))
prec, rec, f1 = confusion2prf(confusion)
print("Prec=%.4f, Rec=%.4f, F1=%.4f" % (prec, rec, f1))