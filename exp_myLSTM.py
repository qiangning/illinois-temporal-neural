import sys

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
from WordEmbeddings_Cache import *
from TemporalDataSet import *
from LemmaEmbeddings import *
from myLSTM import *
from Baseline_LSTM import *
from extract_bigram_stats import temporal_bigram
from pairwise_ffnn_pytorch import VerbNet
from language_model import BieberLSTM

class experiment:
    def __init__(self,model,trainset,testset,output_labels,params,exp_name,modelPath,skiptuning, batch_size):
        self.model = model
        self.params = params
        self.trainset, self.devset = self.split_train_dev(trainset)
        self.testset = testset
        self.output_labels = output_labels
        self.exp_name = exp_name
        self.modelPath = "%s_%s" %(modelPath,self.exp_name)
        self.skiptuning = skiptuning
        self.batch_size = batch_size

    def split_train_dev(self,trainset):
        train,dev = train_test_split(trainset,test_size=0.2,random_state=self.params.get('seed',2093))
        return train,dev
    def train(self):
        self.best_epoch = self.params.get('max_epoch',20)-1
        if not self.skiptuning:
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
        else:
            print("------------Training with the max epoch number (skipped tuning)------------")
        trainset_aug = self.trainset+self.devset
        self.trainHelper(trainset_aug,self.testset,self.best_epoch+1,"retrain")

    def trainHelper(self,trainset,testset,max_epoch,tag):
        self.model.train()
        lr = self.params.get('lr',0.1)
        weight_decay = self.params.get('weight_decay',1e-4)
        step_size = self.params.get('step_size',10)
        gamma = self.params.get('gamma',0.3)
        # optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = optim.Adam(self.model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        lm_embedding_train = open('/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/lm_pair_probabilities_train3.txt')
        lm_embedding_train_lines = lm_embedding_train.readlines()
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
            batch_loss = 0.0
            self.model.zero_grad()
            for i,temprel in enumerate(trainset):
                # self.model.zero_grad()
                target = torch.cuda.LongTensor([self.output_labels[temprel.label]])
                embedding = torch.tensor([[float(x) for x in lm_embedding_train_lines[i].split(' ')]]).cuda()
                output = self.model((temprel, embedding))
                loss = criterion(output, target)
                current_train_loss += loss
                batch_loss += loss
                if i % 1000 == 0:
                    print("%d/%d: %s %.4f %.4f" % (i, len(trainset), timeSince(start), loss, current_train_loss), flush=True)
                if (i+1) % self.batch_size == 0:
                    batch_loss.backward()
                    optimizer.step()
                    self.model.zero_grad()
                    batch_loss = 0.0
            all_train_losses.append(current_train_loss)
            current_train_acc, _ = self.eval(trainset)
            current_test_acc, confusion = self.eval(testset)
            all_train_accuracies.append(float(current_train_acc))
            all_test_accuracies.append(float(current_test_acc))
            print("Loss at epoch %d: %.4f" % (epoch, current_train_loss), flush=True)
            print("Train acc at epoch %d: %.4f" % (epoch, current_train_acc), flush=True)
            print("Dev/Test acc at epoch %d: %.4f" % (epoch, current_test_acc), flush=True)
            print(confusion, flush=True)
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

    def test(self):
        self.model.eval()
        test_acc, test_confusion = self.eval(self.testset)
        print("TEST ACCURACY=%.4f" % test_acc)
        print("CONFUSION MAT:")
        print(test_confusion)

    def eval(self,eval_on_set):
        was_training = self.model.training
        self.model.eval()
        lm_embedding_test = open('/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/lm_pair_probabilities_test3.txt')
        lm_embedding_test_lines = lm_embedding_test.readlines()
        confusion = np.zeros((len(self.output_labels), len(self.output_labels)), dtype=int)
        for ex, line in zip(eval_on_set, lm_embedding_test_lines):
            emb = torch.tensor([[float(x) for x in line.split(' ')]]).cuda()
            output = self.model((ex, emb))
            confusion[self.output_labels[ex.label]][categoryFromOutput(output)] += 1
        if was_training:
            self.model.train()
        return 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion), confusion

    def load_model(self,modelLoadPath):
        print('')

    def logger(self,logPath):
        print('')

class bigramGetter_fromNN:
    def __init__(self,emb_path,mdl_path,ratio=0.3,layer=1,emb_size=200,splitter=','):
        self.verb_i_map = {}
        f = open(emb_path)
        lines = f.readlines()
        for i,line in enumerate(lines):
            self.verb_i_map[line.split(splitter)[0]] = i
        self.model = VerbNet(len(self.verb_i_map),hidden_ratio=ratio,emb_size=emb_size,num_layers=layer)
        checkpoint = torch.load(mdl_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    def eval(self,v1,v2):
        return self.model(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).cuda())
    def getBigramStatsFromTemprel(self,temprel):
        v1,v2='',''
        for i,position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            elif position == 'E2':
                v2 = temprel.lemma[i]
                break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.cuda.FloatTensor([0,0]).view(1,-1)
        return torch.cat((self.eval(v1,v2),self.eval(v2,v1)),1).view(1,-1)
    def retrieveEmbeddings(self,temprel):
        v1, v2 = '', ''
        for i, position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            elif position == 'E2':
                v2 = temprel.lemma[i]
                break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.zeros_like(self.model.retrieveEmbeddings(torch.from_numpy(np.array([[0,0]])).cuda()).view(1,-1))
        return self.model.retrieveEmbeddings(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).cuda()).view(1,-1)

class bigramGetter_fromLM:
    def __init__(self,mdl_path,all_verbs,emb_size=200):
        self.verb_i_map = {}
        for i in range(len(all_verbs)):
            self.verb_i_map[all_verbs[i].decode('utf-8')] = i
        self.model = BieberLSTM(len(all_verbs), nb_layers=1, batch_size=1).cuda()
        checkpoint = torch.load(mdl_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.emb_size = emb_size
    def retrieveEmbeddings(self,temprel):
        v1, v2 = '', ''
        for i, position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            elif position == 'E2':
                v2 = temprel.lemma[i]
                break
        lstm_outputs1 = torch.zeros((1, self.emb_size), dtype=torch.float32)
        lstm_outputs2 = torch.zeros((1, self.emb_size), dtype=torch.float32)
        if v1 in self.verb_i_map:
            hidden = self.model.init_hidden()
            lstm_outputs1, _ = self.model.lstm(self.model.word_embedding(torch.tensor([[self.verb_i_map[v1]]]).cuda()), hidden)
            lstm_outputs1 = lstm_outputs1.view(1, -1)
        if v2 in self.verb_i_map:
            hidden = self.model.init_hidden()
            lstm_outputs2, _ = self.model.lstm(self.model.word_embedding(torch.tensor([[self.verb_i_map[v2]]]).cuda()), hidden)
            lstm_outputs2 = lstm_outputs2.view(1, -1)
        vec1 = torch.zeros((1, self.emb_size), dtype=torch.float32).cuda()
        vec2 = torch.zeros((1, self.emb_size), dtype=torch.float32).cuda()
        if v1 in self.verb_i_map:
            vec1 = self.model.word_embedding(torch.tensor([self.verb_i_map[v1]]).cuda())
        if v2 in self.verb_i_map:
            vec2 = self.model.word_embedding(torch.tensor([self.verb_i_map[v2]]).cuda())
        return torch.cat((lstm_outputs1, lstm_outputs2, vec1, vec2), dim=1)
        """if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.zeros((1, 2), dtype=torch.float32)
        score12 = self.model.forward(torch.tensor([[self.verb_i_map[v1]]]).cuda(), torch.tensor([1]).cuda())[0][0][self.verb_i_map[v2]]
        score21 = self.model.forward(torch.tensor([[self.verb_i_map[v2]]]).cuda(), torch.tensor([1]).cuda())[0][0][self.verb_i_map[v1]]
        return torch.tensor([[score12, score21]]).cuda()"""

@click.command()
@click.option("--w2v_option",default=2)
@click.option("--lstm_hid_dim",default=128)
@click.option("--nn_hid_dim",default=64)
@click.option("--pos_emb_dim",default=32)
@click.option("--common_sense_emb_dim",default=64)
@click.option("--lr",default=0.1)
@click.option("--weight_decay",default=1e-4)
@click.option("--step_size",default=10)
@click.option("--gamma",default=0.2)
@click.option("--max_epoch",default=50)
@click.option("--expname",default="test")
@click.option("--skiptuning", is_flag=True)
@click.option("--mode",default=0)
@click.option("--dropout", is_flag=True)
@click.option("--timeline_kb", is_flag=True)
@click.option("--debug",is_flag=True)
@click.option("--batch_size", default=1)
def run(w2v_option, lstm_hid_dim, nn_hid_dim, pos_emb_dim, common_sense_emb_dim, lr, weight_decay, step_size, gamma, max_epoch, expname, skiptuning, mode, dropout, timeline_kb, debug, batch_size):

    trainset = temprel_set("data/Output4LSTM_Baseline/trainset-temprel.xml")
    testset = temprel_set("data/Output4LSTM_Baseline/testset-temprel.xml")

    if w2v_option == 0:
        embedding_dim = 256
        print("Using ELMo (small)")
        emb_cache = elmo_cache(None, "/shared/preprocessed/qning2/temporal/ser/elmo_cache_small.pkl", verbose=False)
    elif w2v_option == 1:
        embedding_dim = 512
        print("Using ELMo (medium)")
        emb_cache = elmo_cache(None, "/shared/preprocessed/qning2/temporal/ser/elmo_cache_medium.pkl", verbose=False)
    elif w2v_option == 2:
        embedding_dim = 1024
        print("Using ELMo (original)")
        emb_cache = elmo_cache(None, "/shared/preprocessed/qning2/temporal/ser/elmo_cache_original.pkl", verbose=False)
    elif w2v_option == 3:
        embedding_dim = 300
        print("glove.6B.300d-light.magnitude")
        emb_cache = w2v_cache(None, "/shared/preprocessed/qning2/temporal/ser/w2v_cache_glove.6B.300d-light.magnitude.pkl", verbose=False)
    elif w2v_option == 4:
        embedding_dim = 300
        print("glove.6B.300d-medium.magnitude")
        emb_cache = w2v_cache(None, "/shared/preprocessed/qning2/temporal/ser/w2v_cache_glove.6B.300d-medium.magnitude.pkl", verbose=False)
    elif w2v_option == 5:
        embedding_dim = 300
        print("wiki-news-300d-1M-medium.magnitude")
        emb_cache = w2v_cache(None, "/shared/preprocessed/qning2/temporal/ser/w2v_cache_wiki-news-300d-1M-medium.magnitude.pkl", verbose=False)
    elif w2v_option == 6:
        embedding_dim = 300
        print("GoogleNews-vectors-negative300-medium.magnitude")
        emb_cache = w2v_cache(None, "/shared/preprocessed/qning2/temporal/ser/w2v_cache_GoogleNews-vectors-negative300-medium.magnitude.pkl", verbose=False)
    else:
        print("word embedding option is wrong (%d)." % w2v_option)
        embedding_dim = 256
        emb_cache = elmo_cache(None, "/shared/preprocessed/qning2/temporal/ser/elmo_cache_small.pkl", verbose=False)

    position2ix = {"B":0,"M":1,"A":2,"E1":3,"E2":4}
    output_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}

    params = {'embedding_dim':embedding_dim,\
                  'lstm_hidden_dim':lstm_hid_dim,\
                  'nn_hidden_dim':nn_hid_dim,\
                  'position_emb_dim':pos_emb_dim,\
                  'bigramStats_dim':2,\
                  'lemma_emb_dim':2,\
                  'dropout':dropout,\
                  'batch_size':1}
    params_optim = {'lr':lr,'weight_decay':weight_decay,'step_size':step_size,'gamma':gamma,'max_epoch':max_epoch}
    print("___________________HYPER-PARAMETERS:LSTM___________________")
    print(params)
    print("___________________HYPER-PARAMETERS:OPTIMIZER___________________")
    print(params_optim)

    print("MODE=%d" %mode)
    if mode == -1: # Baseline: without position embedding. pure LSTM+NN
        model = lstm_NN_baseline(params, emb_cache)
    elif mode==0: # Proposed baseline: w/ position embedding, but without bigram stats
        model = lstm_NN_position_embedding(params, emb_cache, position2ix)
    elif mode == 1: # Proposed: pairwise, temprob, raw stats
        bigramGetter = pkl.load(open("/shared/preprocessed/qning2/temporal/TemProb/temporal_bigram_stats.pkl", 'rb'))
        model = lstm_NN_bigramStats(params, emb_cache, bigramGetter, position2ix)
    elif mode == 2: # Proposed: pairwise, timelines, raw stats
        bigramGetter=pkl.load(open("/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats_new_samesent.pkl",'rb'))
        model = lstm_NN_bigramStats(params, emb_cache, bigramGetter, position2ix)
    elif mode == 3: # Proposed: pairwise, temprob, nn fitted stats
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        print("ratio=%s,emb_size=%d,layer=%d" %(str(ratio),emb_size,layer))
        emb_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_%.1f_%d_%d_temprob.txt' %(ratio,emb_size,layer)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/pairwise_model_%.1f_%d_%d.pt'%(ratio,emb_size,layer)
        bigramGetter = bigramGetter_fromNN(emb_path,mdl_path,ratio,layer,emb_size)
        model = lstm_NN_bigramStats(params, emb_cache, bigramGetter, position2ix)
    elif mode == 4: # Proposed: pairwised, timelines, nn fitted stats
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        emb_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/embeddings_%.1f_%d_%d_timelines.txt' % (
        ratio, emb_size, layer)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/pairwise_model_%.1f_%d_%d.pt' % (
        ratio, emb_size, layer)
        bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size)
        model = lstm_NN_bigramStats(params, emb_cache, bigramGetter, position2ix)
    elif mode == 5: # Proposed: with embeddings from temprob
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        emb_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_%.1f_%d_%d_temprob.txt' % (
            ratio, emb_size, layer)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/pairwise_model_%.1f_%d_%d.pt' % (
            ratio, emb_size, layer)
        if layer==1:
            params['lemma_emb_dim'] = int(emb_size*2*ratio)
        elif layer == 2:
            params['lemma_emb_dim'] = int(emb_size*2*ratio)+int(emb_size*ratio)
        print("ratio=%s,emb_size=%d,layer=%d,lemma_emb_dim=%d" %(str(ratio),emb_size,layer,params['lemma_emb_dim']))
        bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size,splitter=',')
        model = lstm_NN_embeddings(params, emb_cache, bigramGetter, position2ix)
    elif mode == 6: # Proposed: with embeddings from timelines (pairwise model)
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        emb_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/embeddings_%.1f_%d_%d_timelines.txt' % (
            ratio, emb_size, layer)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/pairwise_model_%.1f_%d_%d.pt' % (
            ratio, emb_size, layer)
        if layer==1:
            params['lemma_emb_dim'] = int(emb_size*2*ratio)
        elif layer == 2:
            params['lemma_emb_dim'] = int(emb_size*2*ratio)+int(emb_size*ratio)
        print("ratio=%s,emb_size=%d,layer=%d,lemma_emb_dim=%d" % (str(ratio), emb_size, layer, params['lemma_emb_dim']))
        bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size,splitter=' ')
        model = lstm_NN_embeddings(params, emb_cache, bigramGetter, position2ix)
    elif mode == 7: # Proposed: with embeddings from temprob but put one extra layer after embeddings before concat with the final layer of output
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        emb_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_%.1f_%d_%d_temprob.txt' % (
            ratio, emb_size, layer)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/pairwise_model_%.1f_%d_%d.pt' % (
            ratio, emb_size, layer)
        if layer==1:
            params['lemma_emb_dim'] = int(emb_size*2*ratio)
        elif layer == 2:
            params['lemma_emb_dim'] = int(emb_size*2*ratio)+int(emb_size*ratio)
        print("ratio=%s,emb_size=%d,layer=%d,lemma_emb_dim=%d" %(str(ratio),emb_size,layer,params['lemma_emb_dim']))
        bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size,splitter=',')
        model = lstm_NN_embeddings2(params, emb_cache, bigramGetter, position2ix)
    elif mode == 8: # mode=7 with dropout for common sense embeddings
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        emb_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_%.1f_%d_%d_temprob.txt' % (
            ratio, emb_size, layer)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/pairwise_model_%.1f_%d_%d.pt' % (
            ratio, emb_size, layer)
        if layer == 1:
            params['lemma_emb_dim'] = int(emb_size * 2 * ratio)
        elif layer == 2:
            params['lemma_emb_dim'] = int(emb_size * 2 * ratio) + int(emb_size * ratio)
        print("ratio=%s,emb_size=%d,layer=%d,lemma_emb_dim=%d" % (str(ratio), emb_size, layer, params['lemma_emb_dim']))
        bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size, splitter=',')
        model = lstm_NN_embeddings3(params, emb_cache, bigramGetter, position2ix)
    elif mode == 9:  # Proposed: pairwise, temprob, raw stats==>categorical common sense embedding
        bigramGetter = pkl.load(open("/shared/preprocessed/qning2/temporal/TemProb/temporal_bigram_stats.pkl", 'rb'))
        model = lstm_NN_bigramStats2(params, emb_cache, bigramGetter, position2ix, granularity=0.1, common_sense_emb_dim=common_sense_emb_dim)
    elif mode == 10: # mode 1 + mode 5
        if timeline_kb:
            bigramGetter = pkl.load(open("/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats_new_all.pkl", 'rb'))
        else: # temprob KB
            bigramGetter = pkl.load(open("/shared/preprocessed/qning2/temporal/TemProb/temporal_bigram_stats.pkl", 'rb'))
        model = lstm_NN_bigramStats3(params, emb_cache, bigramGetter, position2ix, granularity=0.1, common_sense_emb_dim=common_sense_emb_dim)
    elif mode == 11:
        if timeline_kb:
            bigramGetter = pkl.load(open("/shared/preprocessed/qning2/temporal/TimeLines/temporal_bigram_stats_new_all.pkl", 'rb'))
        else: # temprob KB
            bigramGetter = pkl.load(open("/shared/preprocessed/qning2/temporal/TemProb/temporal_bigram_stats.pkl", 'rb'))
        model = lstm_NN_bigramStats5(params, emb_cache, bigramGetter, position2ix, granularity=0.1, common_sense_emb_dim=common_sense_emb_dim)
    elif mode == 12: # NN fitted stats from temprob/timelines, categorical embeddings
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
        if timeline_kb:
            emb_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
            mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
        else:
            emb_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_%.1f_%d_%d_temprob.txt' % (ratio, emb_size, layer)
            mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
        bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size)
        model = lstm_NN_bigramStats3(params, emb_cache, bigramGetter, position2ix, granularity=0.1, common_sense_emb_dim=common_sense_emb_dim)
    elif mode == 13:
        ratio = 0.3
        emb_size = 200
        layer = 1
        print("---------")
        if timeline_kb:
            emb_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
            mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/Timelines/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
        else:
            emb_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/embeddings_%.1f_%d_%d_temprob.txt' % (ratio, emb_size, layer)
            mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/TemProb/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
        if layer == 1:
            params['lemma_emb_dim'] = int(emb_size * 2 * ratio)
        elif layer == 2:
            params['lemma_emb_dim'] = int(emb_size * 2 * ratio) + int(emb_size * ratio)
        print("ratio=%s,emb_size=%d,layer=%d,lemma_emb_dim=%d" % (str(ratio), emb_size, layer, params['lemma_emb_dim']))
        bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size, splitter=',')
        model = lstm_NN_bigramStats4(params, emb_cache, bigramGetter, position2ix, granularity=0.1, common_sense_emb_dim=common_sense_emb_dim)
    elif mode == 14: # Proposed: with embeddings from LM but put one extra layer after embeddings before concat with the final layer of output
        hidden_size = 100
        emb_size = 200
        print("---------")
        all_verbs_path = '/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/all_verbs.npy'
        all_verbs = np.load(all_verbs_path)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/language_model_emb%d_bilstm%d.pt' % (
            emb_size, hidden_size)
        # params['lemma_emb_dim'] = int(emb_size*2)
        bigramGetter = bigramGetter_fromLM(mdl_path, all_verbs, emb_size)
        model = lstm_NN_embeddings2(params, emb_cache, bigramGetter, position2ix)
    elif mode == 15: # Proposed: with embeddings from LM but put one extra layer after embeddings before concat with the final layer of output
        hidden_size = 100
        emb_size = 200
        print("---------")
        all_verbs_path = '/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/all_verbs.npy'
        all_verbs = np.load(all_verbs_path)
        mdl_path = '/shared/preprocessed/sssubra2/embeddings/models/LanguageModel/language_model_emb%d_bilstm%d.pt' % (
            emb_size, hidden_size)
        params['lemma_emb_dim'] = int(emb_size*4)
        bigramGetter = bigramGetter_fromLM(mdl_path, all_verbs, emb_size)
        model = lstm_NN_embeddings3(params, emb_cache, bigramGetter, position2ix)
    else:
        print('Error! No such mode: %d' %mode)
        sys.exit(-1)
    if debug:
        exp = experiment(model=model, trainset=trainset.temprel_ee[:100], testset=testset.temprel_ee[:100], \
                         params=params_optim, exp_name=expname, modelPath="models/ckpt", \
                         output_labels=output_labels, skiptuning=skiptuning,batch_size=batch_size)
    else:
        exp = experiment(model=model,trainset=trainset.temprel_ee,testset=testset.temprel_ee,\
                         params=params_optim,exp_name=expname,modelPath="models/ckpt", \
                         output_labels=output_labels,skiptuning=skiptuning,batch_size=batch_size)
    exp.train()
    exp.test()

if __name__ == '__main__':
    run()
