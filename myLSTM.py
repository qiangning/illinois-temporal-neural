import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class lstm_NN_position_embedding(nn.Module):
    def __init__(self, params,emb_cache,position2ix,bidirectional=False, lowerCase=False):
        super(lstm_NN_position_embedding, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.position_emb_dim = params.get('position_emb_dim',16)
        self.emb_cache = emb_cache
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.bidirectional = bidirectional
        self.lowerCase = lowerCase
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        # self.position_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        if not self.lowerCase:
            embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        else:
            embeddings = self.emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out))
        output = self.h_nn2o(h_nn)
        return output

class lstm_NN_bigramStats(nn.Module):
    def __init__(self, params,emb_cache,bigramGetter,position2ix,bidirectional=False):
        super(lstm_NN_bigramStats, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.position_emb_dim = params.get('position_emb_dim',16)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim+self.bigramStats_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim+self.bigramStats_dim, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        h_nn = F.relu(self.h_lstm2h_nn(torch.cat((lstm_out,bigramstats), 1)))
        output = self.h_nn2o(torch.cat((h_nn,bigramstats),1))
        return output

class lstm_NN_bigramStats2(nn.Module): # convert bigramstats into categorical embeddings
    def __init__(self, params,emb_cache,bigramGetter,position2ix,granularity=0.05, common_sense_emb_dim=64,bidirectional=False):
        super(lstm_NN_bigramStats2, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.position_emb_dim = params.get('position_emb_dim',16)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.position2ix = position2ix
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.common_sense_emb = nn.Embedding(int(1/self.granularity)+1,self.common_sense_emb_dim)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim+self.common_sense_emb_dim, self.nn_hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
        self.common_sense_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor([int(bigramstats[0][0]/self.granularity)])).view(1,-1)
        h_nn = F.relu(self.dropout(self.h_lstm2h_nn(torch.cat((lstm_out,common_sense_emb), 1))))
        output = self.h_nn2o(h_nn)
        return output

class lstm_NN_bigramStats3(nn.Module): # convert bigramstats into categorical embeddings. cat them in the final layer
    def __init__(self, params,emb_cache,bigramGetter,position2ix,granularity=0.05, common_sense_emb_dim=64,bidirectional=False):
        super(lstm_NN_bigramStats3, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.position_emb_dim = params.get('position_emb_dim',16)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.is_dropout = params.get('dropout',False)
        self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.position2ix = position2ix
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.common_sense_emb = nn.Embedding(int(1/self.granularity)+1,self.common_sense_emb_dim)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        if self.is_dropout:
            self.lstm.dropout = 0.5
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        if self.is_dropout:
            self.dropout = nn.Dropout(0.5)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim+self.common_sense_emb_dim, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
        self.common_sense_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor([int(bigramstats[0][0]/self.granularity)])).view(1,-1)
        if self.is_dropout:
            h_nn = F.relu(self.dropout(self.h_lstm2h_nn(lstm_out)))
        else:
            h_nn = F.relu(self.h_lstm2h_nn(lstm_out))
        output = self.h_nn2o(torch.cat((h_nn,common_sense_emb),1))
        return output

class lstm_NN_baseline_bigramStats3(nn.Module): # combination of mode -1 and mode 10
    def __init__(self, params,emb_cache,bigramGetter,granularity=0.05, common_sense_emb_dim=64,bidirectional=False,lowerCase=False):
        super(lstm_NN_baseline_bigramStats3, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.common_sense_emb = nn.Embedding(int(1.0/self.granularity)*self.bigramStats_dim,self.common_sense_emb_dim)
        self.bidirectional = bidirectional
        self.lowerCase = lowerCase
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(2*self.lstm_hidden_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim+self.bigramStats_dim*self.common_sense_emb_dim, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.common_sense_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def forward(self, temprel):
        self.init_hidden()
        if not self.lowerCase:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        else:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
        embeds = embeds.view(temprel.length,self.batch_size,-1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[temprel.event_ix][:][:]
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out.view(1,-1)))

        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor([min(int(1.0/self.granularity)-1,int(bigramstats[0][0]/self.granularity))])).view(1,-1)
        for i in range(1,self.bigramStats_dim):
            tmp = self.common_sense_emb(torch.cuda.LongTensor([(i-1)*int(1.0/self.granularity)+min(int(1.0/self.granularity)-1,int(bigramstats[0][i]/self.granularity))])).view(1,-1)
            common_sense_emb = torch.cat((common_sense_emb,tmp),1)

        output = self.h_nn2o(torch.cat((h_nn,common_sense_emb),1))
        return output

class lstm_NN_bigramStats4(nn.Module): # convert bigramstats into categorical embeddings. extra layer before final layer
    def __init__(self, params,emb_cache,bigramGetter,position2ix,granularity=0.05, common_sense_emb_dim=64,bidirectional=False):
        super(lstm_NN_bigramStats4, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.position_emb_dim = params.get('position_emb_dim',16)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.is_dropout = params.get('dropout',False)
        self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.position2ix = position2ix
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.common_sense_emb = nn.Embedding(int(1/self.granularity)+1,self.common_sense_emb_dim)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        if self.is_dropout:
            self.lstm.dropout = 0.5
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        if self.is_dropout:
            self.dropout = nn.Dropout(0.5)
        self.lemma_emb2h_nn = nn.Linear(self.common_sense_emb_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim*2, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
        self.common_sense_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor([int(bigramstats[0][0]/self.granularity)])).view(1,-1)
        if self.is_dropout:
            h_nn = F.relu(self.dropout(self.h_lstm2h_nn(lstm_out)))
            h_nn2 = F.relu(self.dropout(self.lemma_emb2h_nn(common_sense_emb)))
        else:
            h_nn = F.relu(self.h_lstm2h_nn(lstm_out))
            h_nn2 = F.relu(self.lemma_emb2h_nn(common_sense_emb))
        output = self.h_nn2o(torch.cat((h_nn,h_nn2),1))
        return output

class lstm_NN_bigramStats5(nn.Module): # convert bigramstats into categorical embeddings. extra layer before final layer. dropout
    def __init__(self, params,emb_cache,bigramGetter,position2ix,granularity=0.05, common_sense_emb_dim=64,bidirectional=False):
        super(lstm_NN_bigramStats5, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.position_emb_dim = params.get('position_emb_dim',16)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.position2ix = position2ix
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.common_sense_emb = nn.Embedding(int(1/self.granularity)+1,self.common_sense_emb_dim)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.lemma_emb2h_nn = nn.Linear(self.common_sense_emb_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim*2, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
        self.common_sense_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor([int(bigramstats[0][0]/self.granularity)])).view(1,-1)
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out))
        h_nn2 = F.relu(self.dropout(self.lemma_emb2h_nn(common_sense_emb)))
        output = self.h_nn2o(torch.cat((h_nn,h_nn2),1))
        return output

class lstm_NN_embeddings(nn.Module):
    def __init__(self, params, emb_cache, lemma_emb_cache, position2ix, bidirectional):
        super(lstm_NN_embeddings, self).__init__()
        self.embedding_dim = params.get('embedding_dim')
        self.lemma_emb_dim = params.get('lemma_emb_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.position_emb_dim = params.get('position_emb_dim', 16)

        self.emb_cache = emb_cache
        self.lemma_emb_cache = lemma_emb_cache
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim+self.lemma_emb_dim, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]

        # concat lemma_embeddings to the final layer before output
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out))
        output = self.h_nn2o(torch.cat((h_nn,self.lemma_emb_cache.retrieveEmbeddings(temprel)),1))
        return output

class lstm_NN_embeddings2(nn.Module): # concat differently
    def __init__(self, params, emb_cache, lemma_emb_cache, position2ix):
        super(lstm_NN_embeddings2, self).__init__()
        self.embedding_dim = params.get('embedding_dim')
        self.lemma_emb_dim = params.get('lemma_emb_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.position_emb_dim = params.get('position_emb_dim', 16)

        self.emb_cache = emb_cache
        self.lemma_emb_cache = lemma_emb_cache
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,
                            num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        self.lemma_emb2h_nn = nn.Linear(self.lemma_emb_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim*2, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.lemma_emb2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
    def init_hidden(self):
        self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),
                       torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]

        # concat lemma_embeddings to the final layer before output
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out))
        h_nn2 = F.relu(self.lemma_emb2h_nn(self.lemma_emb_cache.retrieveEmbeddings(temprel)))
        output = self.h_nn2o(torch.cat((h_nn,h_nn2),1)) # concat differently
        return output

class lstm_NN_embeddings3(nn.Module): # dropout
    def __init__(self, params, emb_cache, lemma_emb_cache, position2ix):
        super(lstm_NN_embeddings3, self).__init__()
        self.embedding_dim = params.get('embedding_dim')
        self.lemma_emb_dim = params.get('lemma_emb_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.position_emb_dim = params.get('position_emb_dim', 16)

        self.emb_cache = emb_cache
        self.lemma_emb_cache = lemma_emb_cache
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), self.position_emb_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.position_emb_dim, self.lstm_hidden_dim,
                            num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.lemma_emb2h_nn = nn.Linear(self.lemma_emb_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim*2, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.lemma_emb2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
    def init_hidden(self):
        self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),
                       torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, temprel):
        self.init_hidden()
        embeds = self.temprel2embeddingSeq(temprel)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[-1][:][:]

        # concat lemma_embeddings to the final layer before output
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out))
        h_nn2 = F.relu(self.dropout(self.lemma_emb2h_nn(self.lemma_emb_cache.retrieveEmbeddings(temprel))))
        output = self.h_nn2o(torch.cat((h_nn,h_nn2),1)) # concat differently
        return output

