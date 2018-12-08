import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# obsolete
class lstm_baseline(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, position_emb_dim, emb_cache, output_dim, batch_size, position2ix):
        super(lstm_baseline, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.position_emb_dim = position_emb_dim
        self.emb_cache = emb_cache
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), position_emb_dim)
        self.lstm = nn.LSTM(embedding_dim + position_emb_dim, hidden_dim,
                            num_layers=1, bidirectional=False)
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.init_hidden()

    def init_hidden(self):
        self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.hidden_dim),
                       torch.randn(1 * self.lstm.num_layers, self.batch_size, self.hidden_dim))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, input):
        self.init_hidden()
        # embeds = self.temprel2embeddingSeq(temprel)
        embeds = input
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        output = self.h2o(lstm_out)
        return output


class lstm_NN_baseline(nn.Module):
    def __init__(self, params, emb_cache, bidirectional=False, lowerCase=False):
        super(lstm_NN_baseline, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.emb_cache = emb_cache
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.bidirectional = bidirectional
        self.lowerCase = lowerCase
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim // 2, \
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, \
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(2 * self.lstm_hidden_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim, self.output_dim)
        self.init_hidden()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()

    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2), \
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim), \
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def forward(self, temprel):
        self.init_hidden()
        if not self.lowerCase:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        else:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
        embeds = embeds.view(temprel.length, self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[temprel.event_ix][:][:]
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out.view(1, -1)))
        output = self.h_nn2o(h_nn)
        return output


class NN_baseline(nn.Module):
    def __init__(self, params, emb_cache, bidirectional=False, lowerCase=False):
        super(NN_baseline, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.emb_cache = emb_cache
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.bidirectional = bidirectional
        self.lowerCase = lowerCase
        self.h_word2h_nn = nn.Linear(2 * self.embedding_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim, self.output_dim)

    def reset_parameters(self):
        self.h_word2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()

    def forward(self, temprel):
        if not self.lowerCase:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        else:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
        embeds = embeds.view(temprel.length, self.batch_size, -1)
        h_word = embeds[temprel.event_ix].view(1, -1)
        h_nn = F.relu(self.h_word2h_nn(h_word))
        output = self.h_nn2o(h_nn)
        return output


class lstm_NN_xml(nn.Module):
    def __init__(self, params, emb_cache, bidirectional=False, lowerCase=False):
        super(lstm_NN_xml, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.emb_cache = emb_cache
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.position_emb = nn.Embedding(4, self.embedding_dim)  # <E1>:0 </E1>:1 <E2>:2 </E2>:3
        self.bidirectional = bidirectional
        self.lowerCase = lowerCase
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim // 2, \
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, \
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(self.lstm_hidden_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim, self.output_dim)
        self.init_hidden()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()

    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2), \
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim), \
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def forward(self, temprel):
        self.init_hidden()
        if not self.lowerCase:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        else:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
        embeds = embeds.view(temprel.length, self.batch_size, -1)
        event_ix = temprel.event_ix
        embeds_with_position = torch.cat(
            (embeds[:event_ix[0]][:][:],
             self.position_emb(torch.cuda.LongTensor([0])).view(1, self.batch_size, -1), \
             embeds[event_ix[0]].view(1, self.batch_size, -1), \
             self.position_emb(torch.cuda.LongTensor([1])).view(1, self.batch_size, -1), \
             embeds[event_ix[0] + 1:event_ix[1]][:][:], \
             self.position_emb(torch.cuda.LongTensor([2])).view(1, self.batch_size, -1), \
             embeds[event_ix[1]].view(1, self.batch_size, -1), \
             self.position_emb(torch.cuda.LongTensor([3])).view(1, self.batch_size, -1), \
             embeds[event_ix[1] + 1:][:][:]), 0)
        lstm_out, self.hidden = self.lstm(embeds_with_position, self.hidden)
        lstm_out = lstm_out.view(embeds_with_position.size()[0], self.batch_size, self.lstm_hidden_dim)
        # lstm_out = torch.cat((lstm_out[event_ix[0]+1][:][:],lstm_out[event_ix[1]+3][:][:]),0)
        lstm_out = lstm_out[-1][:][:]
        h_nn = F.relu(self.h_lstm2h_nn(lstm_out.view(1, -1)))
        output = self.h_nn2o(h_nn)
        return output


# obsolete
class bilstm_baseline(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, position_emb_dim, emb_cache, output_dim, batch_size, position2ix):
        super(bilstm_baseline, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.position_emb_dim = position_emb_dim
        self.emb_cache = emb_cache
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.position2ix = position2ix
        self.position_emb = nn.Embedding(len(position2ix), position_emb_dim)
        self.lstm = nn.LSTM(embedding_dim + position_emb_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.init_hidden()

    def init_hidden(self):
        self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.hidden_dim // 2),
                       torch.randn(2 * self.lstm.num_layers, self.batch_size, self.hidden_dim // 2))

    def temprel2embeddingSeq(self, temprel):
        embeddings = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        position_emb = self.position_emb(torch.cuda.LongTensor([self.position2ix[t] for t in temprel.position]))
        return torch.cat((embeddings, position_emb), 1).view(temprel.length, self.batch_size, -1)

    def forward(self, input):
        self.init_hidden()
        # embeds = self.temprel2embeddingSeq(temprel)
        embeds = input
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.hidden_dim)
        lstm_out = lstm_out[-1][:][:]
        output = self.h2o(lstm_out)
        return output
