from exp_myLSTM import *
mydir="/shared/preprocessed/sssubra2/embeddings/models/TemProb/"
ratio = 0.3
layer=1
emb_size=200
emb_path=mydir+"embeddings_%s_%d_%d_temprob.txt" %(str(ratio),emb_size,layer)
mdl_path=mydir+"pairwise_model_%s_%d_%d.pt"%(str(ratio),emb_size,layer)
bigram=bigramGetter_fromNN(emb_path,mdl_path,ratio=ratio,layer=layer,emb_size=emb_size,splitter=',')
def myeval(v1,v2):
    return torch.cat((bigram.eval(v1,v2),bigram.eval(v2,v1)),1)

def show():
    print("debate","vote")
    print(myeval("debate","vote"))
    print("admire","respect")
    print(myeval("admire","respect"))
    print("ask","help")
    print(myeval("ask","help"))
    print("concern","protect")
    print(myeval("concern","protect"))
    print("die","crash")
    print(myeval("die","crash"))
    print("explode","die")
    print(myeval("explode","die"))
    print("defend","accuse")
    print(myeval("defend","accuse"))
    print("attack","die")
    print(myeval("attack","die"))
    print("overthrow","elect")
    print(myeval("overthrow","elect"))
    print("achieve","desire")
    print(myeval("achieve","desire"))
    print("chop","taste")
    print(myeval("chop","taste"))
    print("clean","contaminate")
    print(myeval("clean","contaminate"))
    print("conspire","kill")
    print(myeval("conspire","kill"))
    print("dedicate","promote")
    print(myeval("dedicate","promote"))

if __name__=="__main__":
    show()