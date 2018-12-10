import pickle as pkl
from myFFNN import *
import click
import math

from TemporalDataSet import temprel_set
from exp_myLSTM import experiment
from extract_bigram_stats import temporal_bigram

class TempRel2BigramFeats:
    def __init__(self,bigramGetter):
        self.bigramGetter = bigramGetter
    def temprel2bigram_feats(self,temprel,dry_run=False):
        # cnt(v1,v2), cnt(v2,v1), prob(v1,v2), prob(v2,v1)
        v1, v2 = "", ""
        if not dry_run:
            for i, position in enumerate(temprel.position):
                if position == 'E1':
                    v1 = temprel.lemma[i]
                if position == 'E2':
                    v2 = temprel.lemma[i]
                    break
        ret = [math.log(self.bigramGetter.getBigramCounts(v1,v2)+1),math.log(self.bigramGetter.getBigramCounts(v2,v1)+1)]
        ret+=[math.log(self.bigramGetter.getBigramRelCounts(v1, v2, 'before')+1),math.log(self.bigramGetter.getBigramRelCounts(v2, v1, 'before')+1)]
        ret += [1.0 * (self.bigramGetter.getBigramRelCounts(v1, v2, 'before') + 1) / (
                    self.bigramGetter.getBigramCounts(v1, v2) + 1)]
        ret += [1.0 * (self.bigramGetter.getBigramRelCounts(v2, v1, 'before') + 1) / (
                    self.bigramGetter.getBigramCounts(v2, v1) + 1)]
        return torch.Tensor(ret).view(1, -1)

@click.command()
@click.option("--mode",default=0)
@click.option("--lr",default=0.1)
@click.option("--weight_decay",default=1e-4)
@click.option("--step_size",default=10)
@click.option("--gamma",default=0.2)
@click.option("--max_epoch",default=50)
@click.option("--nn_hidden_dim",default=64)
@click.option("--expname",default="test")
@click.option("--skiptuning", is_flag=True)
@click.option("--skiptraining", is_flag=True)
def run(mode,lr,weight_decay,step_size,gamma,max_epoch,nn_hidden_dim,expname,skiptuning,skiptraining):
    trainset = temprel_set("data/Output4LSTM_Baseline/trainset-temprel.xml")
    testset = temprel_set("data/Output4LSTM_Baseline/testset-temprel.xml")

    bigramGetter = pkl.load(
        open("/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats_new_samesent.pkl", 'rb'))
    temprel2bigram_feats = TempRel2BigramFeats(bigramGetter)

    input_dim = temprel2bigram_feats.temprel2bigram_feats(None,True).size()[1]
    output_labels = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}
    params={"input_dim":input_dim,"nn_hidden_dim":nn_hidden_dim}

    params_optim = {'lr': lr, 'weight_decay': weight_decay, 'step_size': step_size, 'gamma': gamma,
                    'max_epoch': max_epoch}
    print("___________________HYPER-PARAMETERS:NN___________________")
    print(params)
    print("___________________HYPER-PARAMETERS:OPTIMIZER___________________")
    print(params_optim)

    if mode == 0:
        model = KB_Stats_Input2NN(params, temprel2bigram_feats)
    elif mode == 1:
        model = KB_Stats_Input2NN_2layers(params, temprel2bigram_feats)
    exp = experiment(model=model, trainset=trainset.temprel_ee, testset=testset.temprel_ee, \
                     params=params_optim, exp_name=expname, modelPath="models/baseline", \
                     output_labels=output_labels, skiptuning=skiptuning, gen_output=True)
    if not skiptraining:
        exp.train()
    else:
        exp.model.load_state_dict(torch.load(exp.modelPath+"_best")['model_state_dict'])
    exp.test()

if __name__ == "__main__":
    run()