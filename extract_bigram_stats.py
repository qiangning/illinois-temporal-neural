
# coding: utf-8

# In[1]:


import os
import time
import math
import pickle as pkl
from utils import *

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class temporal_bigram:
    def __init__(self):
        self.bigram_rel_counts = {}
        self.bigram_counts = {}
        self.vocab = set()
        self.relationSet = set(['before','after'])
        self.total_counts = 0
        self.total_bigrams = 0
    def save(self, path):
        pkl.dump(self, open(path, 'wb'))
    def addWord(self,word):
        if word not in self.vocab:
            self.vocab.add(word)
    def addRelation(self,rel):
        if rel not in self.relationSet:
            self.relationSet.add(rel)
    def addOneRelation(self,v1,v2,rel,cnt=1):
        self.addWord(v1)
        self.addWord(v2)
        self.addRelation(rel)
        if v1 not in self.bigram_rel_counts:
            self.bigram_rel_counts[v1] = {}
            self.bigram_counts[v1] = {}
        if v2 not in self.bigram_rel_counts[v1]:
            self.bigram_rel_counts[v1][v2] = {}
            self.bigram_counts[v1][v2] = 0
            self.total_bigrams += 1
        if rel not in self.bigram_rel_counts[v1][v2]:
            self.bigram_rel_counts[v1][v2][rel] = 0
        self.bigram_rel_counts[v1][v2][rel] += cnt
        self.bigram_counts[v1][v2] += cnt
        self.total_counts += cnt
    def getBigramCounts(self,v1,v2):
        if v1 not in self.bigram_counts or v2 not in self.bigram_counts[v1]:
            return 0
        return self.bigram_counts[v1][v2]
    def getBigramRelCounts(self,v1,v2,rel):
        if v1 not in self.bigram_rel_counts or v2 not in self.bigram_rel_counts[v1] or rel not in self.bigram_rel_counts[v1][v2]:
            return 0
        return self.bigram_rel_counts[v1][v2][rel]
    def getBigramStatsFromTemprel(self,temprel,isGPU=True):
        v1, v2 = "", ""
        for i,position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            if position == 'E2':
                v2 = temprel.lemma[i]
                break
        ret = []
        if self.getBigramCounts(v1,v2)>0:
            ret.append(1.0 * self.getBigramRelCounts(v1, v2, 'before') / self.getBigramCounts(v1, v2))
        else:
            ret.append(0)
        if self.getBigramCounts(v2,v1)>0:
            ret.append(1.0 * self.getBigramRelCounts(v2, v1, 'before') / self.getBigramCounts(v2, v1))
        else:
            ret.append(0)
        if isGPU:
            return torch.cuda.FloatTensor(ret).view(1,-1)
        else:
            return ret

    def snapshot(self,pairs2monitor=None):
        print("-------------temporal_bigram: basic stats-------------",flush=True)
        print("Length of vocab: %d" % len(self.vocab),flush=True)
        print("Relation set: %s" % str(self.relationSet),flush=True)
        print("Total bigrams added: %d" % self.total_bigrams,flush=True)
        print("Total counts added: %d" % self.total_counts,flush=True)
        print("------------------------------------------------------",flush=True)
        if pairs2monitor is not None:
            print("\t%s\t%s\t%s\t%s" %('Pairs (PBefore)'.ljust(20),'TotalCnt'.ljust(10),'TBefore'.ljust(10),'TAfter'.ljust(10)),flush=True)
            for i, eventPair in enumerate(pairs2monitor):
                v1, v2 = eventPair[0], eventPair[1]
                if v1 not in self.vocab or                 v2 not in self.vocab or                 v1 not in self.bigram_rel_counts or                 v2 not in self.bigram_rel_counts[v1]:
                    continue
                print("\t%s\t%s\t%-10d\t%-5d (%-5.2f%%)\t\t%-5d (%-5.2f%%)" \
                      %(v1.ljust(10),v2.ljust(10),\
                        self.getBigramCounts(v1,v2),\
                        self.getBigramRelCounts(v1,v2,'before'),\
                        100.0*self.getBigramRelCounts(v1,v2,'before')/(self.getBigramCounts(v1,v2)+1e-6),\
                        self.getBigramRelCounts(v1,v2,'after'),\
                        100.0*self.getBigramRelCounts(v1,v2,'after')/(self.getBigramCounts(v1,v2)+1e-6)),\
                      flush=True)
                print("\t%s\t%s\t%-10d\t%-5d (%-5.2f%%)\t\t%-5d (%-5.2f%%)" \
                      %(v2.ljust(10),v1.ljust(10),\
                        self.getBigramCounts(v2,v1),\
                        self.getBigramRelCounts(v2,v1,'before'),\
                        100.0*self.getBigramRelCounts(v2,v1,'before')/(self.getBigramCounts(v2,v1)+1e-6),\
                        self.getBigramRelCounts(v2,v1,'after'),\
                        100.0*self.getBigramRelCounts(v2,v1,'after')/(self.getBigramCounts(v2,v1)+1e-6)),\
                      flush=True)
            for eventPair in pairs2monitor:
                v1, v2 = eventPair[0], eventPair[1]
                if v1 not in self.vocab:
                    print('%s not in vocab!' % v1,flush=True)
                if v2 not in self.vocab:
                    print('%s not in vocab!' % v2,flush=True)
                if v1 not in self.bigram_rel_counts or v2 not in self.bigram_rel_counts[v1]:
                    print('(%s,%s) not in bigrams' % (v1,v2),flush=True)
            print("------------------------------------------------------",flush=True)
            


# In[3]:


class event:
    def __init__(self,lemma,idx):
        self.lemma = lemma
        self.idx = idx
def isPBefore(event1,event2):
    return event1.idx < event2.idx

def extractFromTemprob():
    tBigram = temporal_bigram()
    PAIRS_MONITOR = {('die', 'explode'), ('attack', 'die'), ('ask', 'help'), ('chop', 'taste'), ('concern', 'protect'), \
                     ('conspire', 'kill'), ('debate', 'vote'), ('dedicate', 'promote'), ('fight', 'overthrow'), \
                     ('achieve', 'desire'), ('admire', 'respect'), ('clean', 'contaminate'), ('defend', 'accuse'), \
                     ('die', 'crash'), ('overthrow', 'elect')}
    TEMPROB_DIRECTORY = "/home/qning2/Servers/root/shared/preprocessed/qning2/temporal/temporalLM/sent1results.txt"
    f=open(TEMPROB_DIRECTORY,'r')
    lines = f.readlines()
    print("lines:%d" %len(lines))
    cnt = 0
    for line in lines:
        # cnt += 1
        # if cnt > 20:
        #     break
        tmp = line.split()
        if len(tmp)!=4:
            continue
        if tmp[2] != 'before' and tmp[2] != 'after':
            continue
        v1 = tmp[0][:-3]
        v2 = tmp[1][:-3]
        tBigram.addOneRelation(v1,v2,tmp[2],int(tmp[3]))
    tBigram.snapshot(PAIRS_MONITOR)
    tBigram.save('/home/qning2/Servers/root/shared/preprocessed/qning2/temporal/TemProb/temporal_bigram_stats.pkl')
def extractFromTimelines_new():
    tBigram = temporal_bigram()
    PAIRS_MONITOR = {('die', 'explode'), ('attack', 'die'), ('ask', 'help'), ('chop', 'taste'), ('concern', 'protect'), \
                     ('conspire', 'kill'), ('debate', 'vote'), ('dedicate', 'promote'), ('fight', 'overthrow'), \
                     ('achieve', 'desire'), ('admire', 'respect'), ('clean', 'contaminate'), ('defend', 'accuse'), \
                     ('die', 'crash'), ('overthrow', 'elect')}
    TEMPROB_DIRECTORY = "/home/qning2/Servers/root/shared/preprocessed/qning2/temporal/TempRels-CogCompTime/combined.stats"
    f = open(TEMPROB_DIRECTORY, 'r')
    lines = f.readlines()
    print("lines:%d" % len(lines))
    cnt = 0
    for line in lines:
        # cnt += 1
        # if cnt > 20:
        #     break
        tmp = line.split(',')
        if len(tmp) != 6:
            continue
        if tmp[2].lower() != 'before' and tmp[2].lower() != 'after':
            continue
        v1 = tmp[0]
        v2 = tmp[1]
        tBigram.addOneRelation(v1, v2, tmp[2].lower(), int(tmp[3]))
    tBigram.snapshot(PAIRS_MONITOR)
    tBigram.save('/home/qning2/Servers/root/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats_new_samesent.pkl')
def extractFromTimelines():
    tBigram = temporal_bigram()
    start = time.time()
    MAX_FILES = 2000000
    REPORT_STEP = 100000
    WINDOW = 3
    MIN_LEN_TIMELINE = 5
    PAIRS_MONITOR = {('die', 'explode'), ('attack', 'die'), ('ask', 'help'), ('chop', 'taste'), ('concern', 'protect'), \
                     ('conspire', 'kill'), ('debate', 'vote'), ('dedicate', 'promote'), ('fight', 'overthrow'), \
                     ('achieve', 'desire'), ('admire', 'respect'), ('clean', 'contaminate'), ('defend', 'accuse'), \
                     ('die', 'crash'), ('overthrow', 'elect')}

    # INPUT_DIRECTORY = "/home/qning2/Servers/root/shared/preprocessed/sssubra2/annotated-nyt-temporal/extracted"
    INPUT_DIRECTORY = "/shared/preprocessed/sssubra2/annotated-nyt-temporal/extracted"
    N_FILES = 0
    for root, _, files in os.walk(INPUT_DIRECTORY):
        for fname in files:
            if fname.startswith('._'):
                continue
            infile = os.path.join(root, fname)
            events = []
            try:
                lines = open(infile, 'r').readlines()
            except:
                print("Open %s failed. Skipping." % infile)
                continue
            for line in lines:
                if not line.startswith('E'):
                    continue
                line = line.strip().split("/")
                lemma = line[-2]  # verb lemma
                idx = int(line[0][1:])
                events.append(event(lemma, idx))
            if len(events) < MIN_LEN_TIMELINE:
                continue
            N_FILES += 1
            for i, e1 in enumerate(events):
                for j in range(max(0, i - WINDOW), min(i + WINDOW + 1, len(events))):
                    if i == j:
                        continue
                    e2 = events[j]
                    if isPBefore(e1, e2):  # Text: e1...e2
                        if i < j:  # Time: e1...e2
                            tBigram.addOneRelation(e1.lemma, e2.lemma, 'before')
                        else:  # Time: e2...e1
                            tBigram.addOneRelation(e1.lemma, e2.lemma, 'after')
                    else:  # Text: e2...e1
                        if i < j:  # Time: e1...e2
                            tBigram.addOneRelation(e2.lemma, e1.lemma, 'after')
                        else:  # Time: e2...e1
                            tBigram.addOneRelation(e2.lemma, e1.lemma, 'before')
            if (N_FILES % REPORT_STEP == 0):
                print("%d files have been processed (%s)" % (N_FILES, timeSince(start)), flush=True)
                tBigram.snapshot(PAIRS_MONITOR)
                tBigram.save('temporal_bigram_stats.pkl')
            if (N_FILES > MAX_FILES):
                break
        if (N_FILES > MAX_FILES):
            break

if __name__ == '__main__':
    extractFromTimelines_new()