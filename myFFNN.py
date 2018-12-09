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

    def forward(self, temprel):
        # temprel to input
        input_features = self.temprel2bigram_feats(temprel)
        # input to hidden
        h_nn = F.relu(self.h_input2h_nn(input_features.view(1,-1)))
        # hidden to output
        output = self.h_nn2o(h_nn)
        return output
