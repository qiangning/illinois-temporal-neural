import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class mylstm_NN(nn.Module):
    def __init__(self, embedding_dim, lemma_emb_dim, lstm_hidden_dim, nn_hidden_dim, position_emb_dim, emb_cache, lemma_emb_cache, output_dim,
                 batch_size, position2ix):
        super(mylstm_NN, self).__init__()
        self.embedding_dim = embedding_dim
        self.lemma_emb_dim = lemma_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.position_emb_dim = position_emb_dim
        self.emb_cache = emb_cache
        self.lemma_emb_cache = lemma_emb_cache
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), position_emb_dim)
        self.lstm = nn.LSTM(embedding_dim + position_emb_dim, lstm_hidden_dim,
                            num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(lstm_hidden_dim+2*lemma_emb_dim, nn_hidden_dim)
        self.h_nn2o = nn.Linear(nn_hidden_dim, output_dim)
        self.init_hidden()

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
        E1_lemma_emb = self.lemma_emb_cache.retrieveEmbeddings()
        E2_lemma_emb = self.lemma_emb_cache.retrieveEmbeddings()
        for i in range(temprel.length):
            position = temprel.position[i]
            if position == 'E1':
                E1_lemma_emb = self.lemma_emb_cache.retrieveEmbeddings(temprel.lemma[i])
            elif position == 'E2':
                E2_lemma_emb = self.lemma_emb_cache.retrieveEmbeddings(temprel.lemma[i])
        h_nn = F.relu(self.h_lstm2h_nn(torch.cat((lstm_out,E1_lemma_emb,E2_lemma_emb),1)))
        output = self.h_nn2o(h_nn)
        return output