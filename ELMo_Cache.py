import os.path
import pickle as pkl

from TemporalDataSet import *
from utils import *


class elmo_cache:
    def __init__(self,elmo,cache_path,verbose=False):
        self.elmo = elmo
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
        sentences = [tokList]
        character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)['elmo_representations'][0][0]
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
    trainset = temprel_set("data/Output4LSTM_Baseline/trainset.xml")
    testset = temprel_set("data/Output4LSTM_Baseline/testset.xml")
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    emb_cache = elmo_cache(elmo,"elmo_cache_medium.pkl",True)

    start = time.time()
    for i in range(trainset.size):
        print("%d/%d %s" %(i+1,trainset.size,timeSince(start)))
        temprel = trainset.temprel_ee[i]
        emb_cache.retrieveEmbeddings(temprel.token)
    for i in range(testset.size):
        print("%d/%d %s" %(i+1,testset.size,timeSince(start)))
        temprel = testset.temprel_ee[i]
        emb_cache.retrieveEmbeddings(temprel.token)
    if emb_cache.updated:
        emb_cache.save()