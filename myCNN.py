import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class CNN_bigramStats(nn.Module): # convert bigramstats into categorical embeddings
    def __init__(self, params,emb_cache,bigramGetter,position2ix,granularity=0.05, common_sense_emb_dim=64,bidirectional=False):
        super(CNN_bigramStats, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.cnn_filter_sizes = params.get('cnn_filter_sizes')
        self.cnn_filter_num = params.get('cnn_filter_num')
        self.nn_hidden_dim = params.get('nn_hidden_dim')
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
        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.cnn_filter_num, (s, self.embedding_dim)) for s in self.cnn_filter_sizes])
        self.h_cnn2h_nn = nn.Linear(self.cnn_filter_num*len(self.cnn_filter_sizes)+self.common_sense_emb_dim, self.nn_hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim, self.output_dim)
    def reset_parameters(self):
        self.convs1.reset_parameters()
        self.h_cnn2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
        self.position_emb.reset_parameters()
        self.common_sense_emb.reset_parameters()
    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1) # .view(temprel.length, self.batch_size, -1)
    def forward(self, temprel):
        embeds = self.temprel2embeddingSeq(temprel)
        embeds = embeds.unsqueeze(1)
        conv_res = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        conv_res = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        conv_output = torch.cat(conv_res, 1)
        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        common_sense_emb = self.dropout(self.common_sense_emb(torch.cuda.LongTensor([int(bigramstats[0][0]/self.granularity)])).view(1,-1))
        h_nn = F.relu(self.dropout(self.h_cnn2h_nn(torch.cat((conv_output,common_sense_emb), 1))))
        output = self.h_nn2o(h_nn)
        return output

