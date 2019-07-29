import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class KB_Stats_Input2NN(nn.Module):
    def __init__(self, params, temprel2bigram_feats):
        super(KB_Stats_Input2NN, self).__init__()
        self.params = params
        self.input_dim = params.get('input_dim')
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.temprel2bigram_feats = temprel2bigram_feats
        self.h_input2h_nn = nn.Linear(self.input_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim, self.output_dim)

    def reset_parameters(self):
        self.h_input2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()

    def forward_helper(self,temprel):
        # temprel to input
        input_features = self.temprel2bigram_feats.temprel2bigram_feats(temprel).cuda()
        # input to hidden
        h_nn = F.relu(self.h_input2h_nn(input_features.view(1, -1)))
        return h_nn

    def forward(self, temprel):
        h_nn = self.forward_helper(temprel)
        output = self.h_nn2o(h_nn)
        return output

class KB_Stats_Input2NN_2layers(nn.Module):
    def __init__(self, params, temprel2bigram_feats):
        super(KB_Stats_Input2NN_2layers, self).__init__()
        self.params = params
        self.input_dim = params.get('input_dim')
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.temprel2bigram_feats = temprel2bigram_feats
        self.h_input2h_nn = nn.Linear(self.input_dim, self.nn_hidden_dim)
        self.h_nn2h_nn2 = nn.Linear(self.nn_hidden_dim,32)
        self.h_nn2o = nn.Linear(32, self.output_dim)

    def reset_parameters(self):
        self.h_input2h_nn.reset_parameters()
        self.h_nn2h_nn2.reset_parameters()
        self.h_nn2o.reset_parameters()

    def forward_helper(self,temprel):
        # temprel to input
        input_features = self.temprel2bigram_feats.temprel2bigram_feats(temprel).cuda()
        # input to hidden 1
        h_nn = F.relu(self.h_input2h_nn(input_features.view(1, -1)))
        # hidden 1 to hidden 2
        h_nn2 = F.relu(self.h_nn2h_nn2(h_nn))
        return h_nn2
    def forward(self, temprel):
        h_nn2 = self.forward_helper(temprel)
        output = self.h_nn2o(h_nn2)
        return output
