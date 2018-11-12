"""
This train.py is a modified version of the original pytorch GCN implementation(
    https://github.com/tkipf/pygcn).
    The major modification for this script is adding data_sampling().
"""
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# from pygcn.utils import load_data, accuracy
from utils import load_data, accuracy
from pygcn.models import GCN
from subprocess import call

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# torch.cuda.device(1)
# print('Active CUDA Device: GPU', torch.cuda.current_device())
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test, graph, old_adj, tuples = load_data('/home/feng36/data/dblp/','dblp')
adj, features, labels, idx_train, idx_val, idx_test, graph, old_adj, tuples = load_data('../data/dblp/', 'dblp')

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=features.shape[1]*2,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def data_sampling(features, idx_train, adj, sample_size, neighbor_sample_size, selection=[]):
    """
        1. Selecting edges according to the selection indexs.
        2. For each node, sampling their neighbor nodes.
        3. Build a partial adjacency matrix and neigbor feature vector.
    """
    if len(selection) == 0 : selection = np.random.randint(0, features.shape[0], sample_size)
    concatenated_features = []
    final_features_1 = torch.tensor([])
    final_features_2 = torch.tensor([])
    idx_pair = []
    idx_1 = []
    idx_2 = []
    adj_idx = []

    for idx in selection:
        left, right = tuples[0][idx]
        idx_1.append(left)
        idx_2.append(right)
        idx_pair.append(tuples[0][idx])
        if final_features_1.shape[0] == 0:
            final_features_1 = features[left].view(1,-1)
        else:
            final_features_1 = torch.cat((final_features_1, features[left].view(1,-1)))
        if final_features_2.shape[0] == 0:
            final_features_2 = features[right].view(1,-1)
        else:
            final_features_2 = torch.cat((final_features_2, features[right].view(1,-1)))

    final_l = idx_1 + idx_2 # this is 2n
    sampled_neighbors = []
    col_dim = 0
    dim_check = 0
    final_input_features = torch.tensor([])
    for idx in final_l:
        if idx not in graph:
            sampled_neighbors.append([idx])
            if final_input_features.shape[0] == 0 :
                final_input_features = features[idx].view(1,-1)
            else: final_input_features = torch.cat((final_input_features, features[idx].view(1,-1)))
            col_dim+=1
            dim_check+=1
        else:
            if len(graph[idx]) <= neighbor_sample_size:
                sampled_neighbors.append([idx] + graph[idx])
                if final_input_features.shape[0] == 0 :
                    final_input_features = features[idx].view(1,-1)
                else:
                    final_input_features = torch.cat((final_input_features, features[idx].view(1,-1)))
                final_input_features = torch.cat((final_input_features,features[graph[idx]]))
                col_dim+= (1+features[graph[idx]].shape[0])
                dim_check+=(1+features[graph[idx]].shape[0])
            else:
                idx_for_sample = np.random.randint(0,len(graph[idx]), neighbor_sample_size)
                sample_set = [graph[idx][x_j] for x_j in range(len(graph[idx])) if x_j in set(idx_for_sample)]
                sampled_neighbors.append([idx] + sample_set)
                if final_input_features.shape[0] == 0 :
                    final_input_features = features[idx].view(1,-1)
                else: final_input_features = torch.cat((final_input_features, features[idx].view(1,-1)))
                final_input_features = torch.cat((final_input_features, features[sample_set]))
                dim_check+=(1+features[sample_set].shape[0])
                col_dim+= (1+features[sample_set].shape[0])
    sampled_adj = np.zeros((len(final_l), col_dim))
    col_idx = 0
    for row_idx in range(len(final_l)):
        for real_col_idx in sampled_neighbors[row_idx]:
            sampled_adj[row_idx, col_idx] = old_adj[final_l[row_idx], real_col_idx]
            col_idx += 1

    sampled_labels = torch.cat((final_features_1, final_features_2), dim = 1)
    return final_input_features, sampled_adj, sampled_labels, torch.nonzero(torch.sum(sampled_labels, 1)).shape[0]


def train(epoch, sample_size, neighbor_sample_size):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    nezro_count = 0
    for itr in range(10000):
        samepled_features, sampled_adj, sampled_labels, nzero = data_sampling(features, idx_train, adj, sample_size, neighbor_sample_size,range(int(itr*tuples[0].shape[0]/10000), min(int((itr+1) *tuples[0].shape[0]/10000), tuples[0].shape[0])))
        samepled_features = samepled_features.cuda()
        sampled_labels = sampled_labels.cuda()
        print(samepled_features.shape)
        # sampled_adj = sampled_adj.cuda()
        nezro_count+=nzero
        if samepled_features.shape[0] == 0: return
        sampled_adj= torch.from_numpy(sampled_adj).cuda()
        # sampled_adj.cuda()
        output = model(samepled_features, sampled_adj)
        loss_train = F.mse_loss(output, sampled_labels.double())
        print("Process: %d/10000 with loss %f"% (itr, loss_train))
        # acc_train = accuracy(output, sampled_labels)
        # print("Train loss:", loss_train)
        loss_train.backward()
        optimizer.step()
    print("[Epoch]Nonzero feature count:", nezro_count)
    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)
    #
    # loss_val = F.mse_loss(output, sampled_labels.double())
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          # 'acc_train: {:.4f}'.format(acc_train.item()),
          # 'loss_val: {:.4f}'.format(loss_val.item()),
          # 'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch, 100, 10)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
