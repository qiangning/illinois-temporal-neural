from TemporalDataSet import temprel_set
import numpy as np
from utils import confusion2prf

if __name__=='__main__':
    testset = temprel_set("data/Output4LSTM_Baseline/testset-temprel.xml")
    # f = open('/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/lm_pair_probabilities_test.txt')
    f = open('/home/qning2/Servers/root/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/lm_pair_probabilities_test3.txt')
    lines = f.readlines()
    confusion = np.zeros((4, 4), dtype=int)
    label2ix = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}
    for i,line in enumerate(lines):
        parts = line.split(' ')
        prob12 = float(parts[0])
        prob21 = float(parts[1])
        if prob12 >= prob21:
            pred = 0
        else:
            pred = 1
        gold = label2ix[testset.temprel_ee[i].label]
        print(gold,pred)
        confusion[gold][pred] += 1
    print(confusion)
    print("Acc")
    print(1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion))
    prec, rec, f1 = confusion2prf(confusion)
    print("Prec=%.4f, Rec=%.4f, F1=%.4f" % (prec, rec, f1))