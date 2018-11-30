import matplotlib
import click
from scipy.signal import savgol_filter

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from utils import *
seed_everything(13234)

from ELMo_Cache import *
from TemporalDataSet import *
from LemmaEmbeddings import *
from myLSTM import *
from Baseline_LSTM import *
from extract_bigram_stats import temporal_bigram


class experiment:
    def __init__(self,model,trainset,testset,output_labels,params,exp_name,modelPath):
        self.model = model
        self.params = params
        self.trainset, self.devset = self.split_train_dev(trainset)
        self.testset = testset
        self.output_labels = output_labels
        self.exp_name = exp_name
        self.modelPath = "%s_%s" %(modelPath,self.exp_name)

    def split_train_dev(self,trainset):
        train,dev = train_test_split(trainset,test_size=0.2,random_state=self.params.get('seed',2093))
        return train,dev
    def train(self):
        print("------------Training and Development------------")
        all_train_losses, all_train_accuracies, all_test_accuracies =\
            self.trainHelper(self.trainset,self.devset,self.params.get('max_epoch',20),"tuning")
        all_test_accuracies_smooth = savgol_filter(all_test_accuracies, 3, 1)
        self.best_epoch, best_dev_acc = 0,0
        for i,acc in enumerate(all_test_accuracies_smooth):
            if acc > best_dev_acc:
                best_dev_acc = acc
                self.best_epoch = i

        print("Best epoch=%d, best_dev_acc=%.4f/%.4f (before/after smoothing)" \
              % (self.best_epoch + 1, all_test_accuracies[self.best_epoch], all_test_accuracies_smooth[self.best_epoch]))

        print("------------Training with the best epoch number------------")
        trainset_aug = self.trainset+self.devset
        self.trainHelper(trainset_aug,self.testset,self.best_epoch+1,"retrain")

    def trainHelper(self,trainset,testset,max_epoch,tag):
        lr = self.params.get('lr',0.1)
        weight_decay = self.params.get('weight_decay',1e-4)
        step_size = self.params.get('step_size',10)
        gamma = self.params.get('gamma',0.3)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        criterion = nn.CrossEntropyLoss()
        all_train_losses = []
        all_train_accuracies = []
        all_test_accuracies = []
        start = time.time()
        self.model.reset_parameters()
        for epoch in range(max_epoch):
            print("epoch: %d" % epoch, flush=True)
            current_train_loss = 0
            random.shuffle(trainset)
            scheduler.step()
            for i,temprel in enumerate(trainset):
                self.model.zero_grad()
                target = torch.cuda.LongTensor([self.output_labels[temprel.label]])
                output = self.model(temprel)
                loss = criterion(output, target)
                current_train_loss += loss
                if i % 1000 == 0:
                    print("%d/%d: %s %.4f %.4f" % (i, len(trainset), timeSince(start), loss, current_train_loss), flush=True)
                loss.backward()
                optimizer.step()
            all_train_losses.append(current_train_loss)
            current_train_acc, _ = self.eval(trainset)
            current_test_acc, _ = self.eval(testset)
            all_train_accuracies.append(float(current_train_acc))
            all_test_accuracies.append(float(current_test_acc))
            print("Loss at epoch %d: %.4f" % (epoch, current_train_loss), flush=True)
            print("Train acc at epoch %d: %.4f" % (epoch, current_train_acc), flush=True)
            print("Dev/Test acc at epoch %d: %.4f" % (epoch, current_test_acc), flush=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss
            }, self.modelPath)
            # plot figures
            plt.figure(figsize=(6,6))
            plt.subplot(211)
            plt.plot(all_train_losses,'k')
            plt.grid()
            plt.ylabel('Training loss')
            plt.xlabel('Epoch')
            plt.rcParams.update({'font.size': 12})
            plt.subplot(212)
            plt.plot(all_train_accuracies,'k--')
            plt.plot(all_test_accuracies,'k-*')
            plt.legend(["Train","Dev/Test"])
            plt.grid()
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.rcParams.update({'font.size': 12})

            plt.tight_layout()
            plt.savefig("figs/%s_%s.pdf" % (self.exp_name,tag))
            plt.savefig("figs/%s_%s.pdf" % (self.exp_name,tag))
            plt.close('all')
        return all_train_losses,all_train_accuracies,all_test_accuracies


    def eval(self,eval_on_set):
        confusion = np.zeros((len(self.output_labels), len(self.output_labels)), dtype=int)
        for ex in eval_on_set:
            output = self.model(ex)
            confusion[self.output_labels[ex.label]][categoryFromOutput(output)] += 1
        return 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion), confusion

    def load_model(self,modelLoadPath):
        print('')

    def logger(self,logPath):
        print('')


@click.command()
@click.option("--elmo_option",default='medium')
@click.option("--lstm_hid_dim",default=128)
@click.option("--nn_hid_dim",default=64)
@click.option("--pos_emb_dim",default=32)
@click.option("--lr",default=0.1)
@click.option("--weight_decay",default=1e-4)
@click.option("--step_size",default=10)
@click.option("--gamma",default=0.2)
@click.option("--max_epoch",default=50)
@click.option("--expname",default="test")
def run(elmo_option,lstm_hid_dim,nn_hid_dim,pos_emb_dim,lr,weight_decay,step_size,gamma,max_epoch,expname):

    trainset = temprel_set("data/Output4LSTM_Baseline/trainset.xml")
    testset = temprel_set("data/Output4LSTM_Baseline/testset.xml")

    if elmo_option == 'small':
        embedding_dim = 256
    elif elmo_option == 'medium':
        embedding_dim = 512
    elif elmo_option == 'original':
        embedding_dim = 1024
    else:
        print("elmo option is wrong (%s). has to be small/medium/original" % elmo_option)
        elmo_option = 'small'
        embedding_dim = 256
    emb_cache = elmo_cache(None,"ser/elmo_cache_%s.pkl"%elmo_option,verbose=False)

    position2ix = {"B":0,"M":1,"A":2,"E1":3,"E2":4}
    output_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}

    params = {'embedding_dim':embedding_dim,\
                  'lstm_hidden_dim':lstm_hid_dim,\
                  'nn_hidden_dim':nn_hid_dim,\
                  'position_emb_dim':pos_emb_dim,\
                  'bigramStats_dim':2,\
                  'lemma_emb_dim':200,\
                  'batch_size':1}
    params_optim = {'lr':lr,'weight_decay':weight_decay,'step_size':step_size,'gamma':gamma,'max_epoch':max_epoch}
    print("___________________HYPER-PARAMETERS:LSTM___________________")
    print(params)
    print("___________________HYPER-PARAMETERS:OPTIMIZER___________________")
    print(params_optim)
    # Baseline: without bigram stats
    # model = lstm_NN_baseline(params, emb_cache, position2ix)

    # Proposed: with bigram stats from timelines only
    bigramGetter=pkl.load(open("/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats.pkl",'rb'))
    model = lstm_NN_bigramStats(params, emb_cache, bigramGetter, position2ix)

    # Proposed: with embeddings from temprob
    # lemma_emb_cache = lemmaEmbeddings('ser/embeddings_0.3_200_1_temprob.pkl')
    # model = lstm_NN_embeddings(params, emb_cache, lemma_emb_cache, position2ix)

    exp = experiment(model=model,trainset=trainset.temprel_ee,testset=testset.temprel_ee,\
                     params=params_optim,exp_name=expname,modelPath="models/ckpt", \
                     output_labels=output_labels)
    exp.train()
    test_acc, test_confusion = exp.eval(testset.temprel_ee)
    print("TEST ACCURACY=%.2f" % test_acc)
    print("CONFUSION MAT:")
    print(test_confusion)

if __name__ == '__main__':
    run()