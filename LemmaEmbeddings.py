import pickle as pkl
import torch

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class lemmaEmbeddings:
    def __init__(self,path2pkl,defaultDim=200):
        self.path2pkl = path2pkl
        self.embeddings = pkl.load(open(path2pkl,'rb'))
        self.defaultDim = defaultDim
    def retrieveEmbeddings(self,word=None):
        if word is None or word not in self.embeddings:
            return torch.cuda.FloatTensor(1, self.defaultDim).fill_(0)
        return torch.cuda.FloatTensor(self.embeddings[word]).view(1,-1)

if __name__ =='__main__':

    fname = "/home/qning2/Servers/root/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_0.3_200_1_temprob.txt"
    emb = {}
    with open(fname,"r") as f:
        allLines = f.readlines()
        for line in allLines:
            tmp = line.split(',')
            word = tmp[0]
            vec = [float(x) for x in tmp[1:]]
            emb[word] = vec
    print(len(emb))
    pkl.dump(emb,open("ser/embeddings_0.3_200_1_temprob.pkl","wb"))
    # tmp = lemmaEmbeddings('LemmaEmbeddings_cache.pkl')
    # vec = tmp.retrieveEmbeddings('aaaa')
    # print(vec)
    # print(len(vec))