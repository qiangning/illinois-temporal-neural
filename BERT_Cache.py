import os.path
import pickle as pkl

from TemporalDataSet import *
from utils import *
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


class bert_cache:
    def __init__(self,bert,tokenizer,cache_path,verbose=False):
        self.bert = bert
        self.cache_path = cache_path
        self.verbose = verbose
        self.load()
        self.updated = False
        self.tokenizer = tokenizer

    def load(self):
        # if exists
        if os.path.isfile(self.cache_path):
            self.cache = pkl.load(open(self.cache_path,"rb"))
        else:
            self.cache = {}
    def save(self):
        pkl.dump(self.cache,open(self.cache_path,"wb"))
    def tokList2str(self,tokList):
        return str([x.strip() for x in tokList])
    def add2cache(self,tokList,embeddings):
        cachekey = self.tokList2str(tokList)
        if cachekey not in self.cache:
            self.cache[cachekey] = embeddings
    def process(self,tokList):
        tmp = []
        for tok in tokList:
            tmp2 = self.tokenizer.tokenize(tok)
            if len(tmp2)==0:
                tmp.append("[MASK]")
            else:
                tmp.append(tmp2[0])
        tokList = tmp
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokList)
        tokens_tensor = torch.LongTensor([indexed_tokens])
        embeddings, _ = self.bert(tokens_tensor,output_all_encoded_layers=False)
        embeddings = embeddings.view(len(tokList),-1)
        return embeddings
    def retrieveEmbeddings(self,tokList):
        cachekey = self.tokList2str(tokList)
        if cachekey in self.cache:
            if(self.verbose):
                print("Sentence exists in cache.")
            return self.cache[cachekey]
        if(self.verbose):
            print("Sentence doesn't exist in cache. Processing it.")
        embedding = self.process(tokList)
        self.updated = True
        self.add2cache(tokList,embedding)
        return embedding



if __name__ == "__main__":
    trainset = temprel_set("data/Output4LSTM_Baseline/trainset-temprel.xml")
    testset = temprel_set("data/Output4LSTM_Baseline/testset-temprel.xml")
    bert = BertModel.from_pretrained('bert-large-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert.eval()
    emb_cache = bert_cache(bert,tokenizer,"ser/bert_large_uncased_cache.pkl",True)

    start = time.time()
    # for i in range(trainset.size)):
    for i in range(trainset.size):
        print("%d/%d %s" %(i+1,trainset.size,timeSince(start)))
        temprel = trainset.temprel_ee[i]
        emb_cache.retrieveEmbeddings([x.lower() for x in temprel.token])
    for i in range(testset.size):
        print("%d/%d %s" %(i+1,testset.size,timeSince(start)))
        temprel = testset.temprel_ee[i]
        emb_cache.retrieveEmbeddings([x.lower() for x in temprel.token])
    if emb_cache.updated:
        emb_cache.save()