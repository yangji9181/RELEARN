"""
Added Fully Connected Layer.
"""
import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # print(adj.double(), support.double())
        output = torch.spmm(adj.double() , support.double())
        if self.bias is not None:
            return output + self.bias.double()
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FullyConnectedLayer(Module):
    """
    Simple FC layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(FullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = int(out_features/2)
        self.weight = Parameter(torch.FloatTensor(in_features, int(out_features/2)))
        if bias:
            self.bias = Parameter(torch.FloatTensor(int(out_features/2)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.mm(input, self.weight.double())
        if self.bias is not None:
            support += self.bias.double()
        splitted_outputs = torch.split(support,int(support.shape[0]/2))
        print(splitted_outputs[0].shape, splitted_outputs[1].shape, support.shape)
        output = torch.cat((splitted_outputs[0],splitted_outputs[1]), dim = 1)
        print("final layer output shape",output.shape)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
