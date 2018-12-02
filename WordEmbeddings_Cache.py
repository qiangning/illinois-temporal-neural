import os.path
import pickle as pkl
from pymagnitude import *
from os.path import isfile, join

from TemporalDataSet import *
from utils import *
import click

class w2v_cache:
    def __init__(self,magnitude,cache_path,verbose=False):
        self.magnitude = magnitude
        self.cache_path = cache_path
        self.verbose = verbose
        self.load()
        self.updated = False
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
        embeddings = []
        for tok in tokList:
            embeddings.append(self.magnitude.query(tok))
        return torch.Tensor(embeddings)
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



@click.command()
@click.option("--embdir",default='/home/qning2/Servers/root/shared/preprocessed/qning2/MagnitudeEmbeddings')
@click.option("--embname",default='glove.6B.300d-medium.magnitude')
def run(embdir,embname):
    trainset = temprel_set("data/Output4LSTM_Baseline/trainset.xml")
    testset = temprel_set("data/Output4LSTM_Baseline/testset.xml")

    magnitude = Magnitude(join(embdir, embname))
    emb_cache = w2v_cache(magnitude, "ser/w2v_cache_%s.pkl"%embname, True)

    start = time.time()
    for i in range(trainset.size):
        print("%d/%d %s" % (i + 1, trainset.size, timeSince(start)))
        temprel = trainset.temprel_ee[i]
        emb_cache.retrieveEmbeddings(temprel.token)
    for i in range(testset.size):
        print("%d/%d %s" % (i + 1, testset.size, timeSince(start)))
        temprel = testset.temprel_ee[i]
        emb_cache.retrieveEmbeddings(temprel.token)
    if emb_cache.updated:
        emb_cache.save()
if __name__ == "__main__":
    run()
    # run(embname='wiki-news-300d-1M-medium.magnitude')
    # run(embname='GoogleNews-vectors-negative300-medium.magnitude')
    # magnitude = Magnitude(join("/home/qning2/Servers/root/shared/preprocessed/qning2/MagnitudeEmbeddings", 'glove.6B.300d-light.magnitude'))
    # emb_cache = w2v_cache(magnitude, "ser/w2v_cache_%s.pkl" % 'glove.6B.300d-medium.magnitude', True)
    # emb_cache.retrieveEmbeddings(["I"])