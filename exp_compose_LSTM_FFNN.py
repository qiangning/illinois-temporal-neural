import pickle as pkl

from BERT_Cache import bert_cache
from Baseline_LSTM import lstm_NN_baseline
from exp_myFFNN import TempRel2BigramFeats
from myFFNN import *
import click
import math

from TemporalDataSet import temprel_set
from exp_myLSTM import experiment
from extract_bigram_stats import temporal_bigram

class composed_lstm_ffnn(nn.Module):
    def __init__(self,param,model_lstm,model_ffnn):
        super(composed_lstm_ffnn, self).__init__()
        self.model_lstm = model_lstm
        self.model_ffnn = model_ffnn
        self.input_dim = param.get("input_dim")
        self.output = nn.Linear(self.input_dim,4)
        self.isFinetuning = False
        # init weight
        #self.output.weight = torch.cat((self.model_lstm.h_nn2o.weight,self.model_ffnn.h_nn2o.weight),1)

    def setFinetune(self,isFinetuning):
        self.isFinetuning = isFinetuning

    def reset_parameters(self):
        #self.input2hid.reset_parameters()
        self.output.reset_parameters()

    def forward(self,temprel):
        if self.isFinetuning:
            input1 = self.model_lstm.forward_helper(temprel)
            input2 = self.model_ffnn.forward_helper(temprel)
        else:
            with torch.no_grad():
                input1 = self.model_lstm.forward_helper(temprel)
                input2 = self.model_ffnn.forward_helper(temprel)
        h_input = torch.cat((input1,input2),1)
        output = self.output(h_input)
        return output

@click.command()
@click.option("--mode",default=0)
@click.option("--lr",default=0.1)
@click.option("--weight_decay",default=1e-4)
@click.option("--step_size",default=10)
@click.option("--gamma",default=0.2)
@click.option("--max_epoch",default=50)
@click.option("--finetune", default=0)
@click.option("--expname",default="test")
@click.option("--skiptuning", is_flag=True)
@click.option("--skiptraining", is_flag=True)
def run(mode,lr,weight_decay,step_size,gamma,max_epoch,finetune,expname,skiptuning,skiptraining):
    trainset = temprel_set("data/Output4LSTM_Baseline/trainset-temprel.xml")
    testset = temprel_set("data/Output4LSTM_Baseline/testset-temprel.xml")

    # load ffnn
    bigramGetter = pkl.load(
        open("/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats_new_samesent.pkl", 'rb'))
    temprel2bigram_feats = TempRel2BigramFeats(bigramGetter)

    input_dim = temprel2bigram_feats.temprel2bigram_feats(None,True).size()[1]
    output_labels = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}
    params_ffnn={"input_dim":input_dim,"nn_hidden_dim":128}

    model_ffnn = KB_Stats_Input2NN_2layers(params_ffnn, temprel2bigram_feats)
    model_ffnn.eval()

    # load lstm
    params_lstm = {'embedding_dim': 1024, \
              'lstm_hidden_dim': 64, \
              'nn_hidden_dim': 64}
    w2v_option = 7
    print("bert_large_uncased_cache")
    emb_cache = bert_cache(None, None, "ser/bert_large_uncased_cache.pkl", verbose=False)
    model_lstm = lstm_NN_baseline(params_lstm, emb_cache, lowerCase=w2v_option == 7)
    model_lstm.eval()

    # compose
    params = {"input_dim":params_lstm.get("nn_hidden_dim")+32}
    model_compose = composed_lstm_ffnn(params,model_lstm,model_ffnn)

    # optimizer
    params_optim = {'lr': lr, 'weight_decay': weight_decay, 'step_size': step_size, 'gamma': gamma,
                    'max_epoch': max_epoch, "finetune":finetune}
    print("___________________HYPER-PARAMETERS:LSTM___________________")
    print(params_lstm)
    print("___________________HYPER-PARAMETERS:NN___________________")
    print(params_ffnn)
    print("___________________HYPER-PARAMETERS:OPTIMIZER___________________")
    print(params_optim)

    exp = experiment(model=model_compose, trainset=trainset.temprel_ee, testset=testset.temprel_ee, \
                     params=params_optim, exp_name=expname, modelPath="models/compose_naive", \
                     output_labels=output_labels, skiptuning=skiptuning, gen_output=True)
    if not skiptraining:
        exp.train()
    else:
        exp.model.load_state_dict(torch.load(exp.modelPath+"_best")['model_state_dict'])
    exp.test()

if __name__ == "__main__":
    run()