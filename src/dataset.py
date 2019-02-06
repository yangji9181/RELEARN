from __future__ import division
from __future__ import print_function

import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import load_data


class EvaDataset(Dataset):
    def __init__(self, X, y):
        self.len = len(X)
        self.data = [(X[i], y[i]) for i in range(self.len)]

    def __getitem__(self, index):
        batch, label = self.data[index]
        return batch, label

    def __len__(self):
        return self.len


class Dataset:
    def __init__(self, args, dataset):
        self.adj, self.features, self.graph, self.old_adj, self.links, self.nonzero_nodes, self.diffusion \
            = load_data(f'data/{dataset}/', dataset, args.diffusion_threshold)

        self.num_node, self.feature_len = self.features.shape
        self.content_len = len(self.diffusion[0][1])
        self.num_link = len(self.links)
        self.num_diff = len(self.diffusion)
        self.num_label = args.output_dim

        self.neighbor_sample_size = args.neighbor_sample_size
        self.sample_size = args.sample_size
        self.negative_sample_size = args.negative_sample_size

        sample_size = int(self.sample_size / (1 + self.negative_sample_size))
        self.binary_label = torch.FloatTensor([1]*sample_size + [0]*self.negative_sample_size*sample_size).view(-1, 1)

        self.use_superv = args.use_superv
        if args.use_superv:
            self.superv_sample_size = int(args.t*self.sample_size)
            self.sample_size = self.sample_size-self.superv_sample_size
            self.prior, ls = {}, []
            for (d1, d2, l) in args.train:
                d1, d2 = int(d1), int(d2)
                ls += [(d1, d2), (d2, d1)]
                self.prior[(d1, d2)] = l
                self.prior[(d2, d1)] = l
            self.superv = self.build_superv_dict()
            self.superv['node'] = np.array(ls)
            self.num_superv_node = len(ls)
            self.num_superv_link = len(self.superv['link'])

    def set_superv_sample_ratio(self, t):
        if t > 0:
            sample_size = self.sample_size + self.superv_sample_size
            self.superv_sample_size = int(t * sample_size)
            self.sample_size = sample_size - self.superv_sample_size

    def build_superv_dict(self):
        superv = {'link':[], 'diffusion':{}}
        superv['link'] = np.array([(s, c) for (s, c) in self.links if (s, c) in self.prior])
        for i, (diff, _) in enumerate(self.diffusion):
            l = []
            for j in diff:
                for t in diff:
                    if j!=t and (j, t) in self.prior:
                        l.append((j, t))
            superv['diffusion'][i] = np.array(l)
        return superv

    def sample_subgraph(self, selection, generate_prior=True, return_nodes=False):
        """
            1. Selecting edges according to the selection indexs.
            2. For each node, sampling their neighbor nodes.
            3. Build a partial adjacency matrix and neigbor feature vector.
        """

        if isinstance(selection, list):
            if isinstance(selection[0], list):
                final_l = [i for l in selection for i in l]
            else:
                final_l = selection
        elif isinstance(selection, np.ndarray):
            final_l = selection.flatten()
        else:
            exit('unknown type for selection')

        if hasattr(self, 'prior') and generate_prior:
            prior = np.ones((int(len(final_l)/2), self.num_label)) * 1/self.num_label
            for idx, i in enumerate(range(0, len(final_l), 2)):
                d1, d2 = final_l[i:i+2]
                if (d1, d2) in self.prior:
                    prior[idx] = 0
                    prior[idx, self.prior[(d1, d2)]] = 1
            prior = torch.FloatTensor(prior)
        else:
            prior = None

        sampled_neighbors = []
        col_dim = 0
        # dim_check = 0
        final_input_features = torch.tensor([])
        for idx in final_l:
            if idx not in self.graph:
                sampled_neighbors.append([idx])
                if final_input_features.shape[0] == 0 :
                    final_input_features = self.features[idx].view(1,-1)
                else: final_input_features = torch.cat((final_input_features, self.features[idx].view(1,-1)))
                col_dim+=1
                # dim_check+=1
            else:
                if len(self.graph[idx]) <= self.neighbor_sample_size:
                    sampled_neighbors.append([idx] + self.graph[idx])
                    if final_input_features.shape[0] == 0 :
                        final_input_features = self.features[idx].view(1,-1)
                    else:
                        final_input_features = torch.cat((final_input_features, self.features[idx].view(1,-1)))
                    final_input_features = torch.cat((final_input_features, self.features[self.graph[idx]]))
                    col_dim+= (1+self.features[self.graph[idx]].shape[0])
                    # dim_check+=(1+self.features[self.graph[idx]].shape[0])
                else:
                    idx_for_sample = np.random.randint(0,len(self.graph[idx]), self.neighbor_sample_size)
                    sample_set = [self.graph[idx][x_j] for x_j in range(len(self.graph[idx])) if x_j in set(idx_for_sample)]
                    sampled_neighbors.append([idx] + sample_set)
                    if final_input_features.shape[0] == 0 :
                        final_input_features = self.features[idx].view(1,-1)
                    else: final_input_features = torch.cat((final_input_features, self.features[idx].view(1,-1)))
                    final_input_features = torch.cat((final_input_features, self.features[sample_set]))
                    col_dim+= (1+self.features[sample_set].shape[0])
                    # dim_check+=(1+self.features[sample_set].shape[0])
        sampled_adj = np.zeros((len(final_l), col_dim))
        col_idx = 0
        for row_idx in range(len(final_l)):
            for real_col_idx in sampled_neighbors[row_idx]:
                sampled_adj[row_idx, col_idx] = self.old_adj[final_l[row_idx], real_col_idx]
                col_idx += 1

        # sampled_labels = torch.cat((final_features_1, final_features_2), dim = 1)
        # return final_input_features, sampled_adj, sampled_labels, torch.nonzero(torch.sum(sampled_labels, 1)).shape[0]
        if return_nodes:
            return final_input_features, torch.from_numpy(sampled_adj).float(), prior, final_l
        else:
            return final_input_features, torch.from_numpy(sampled_adj).float(), prior

    def sample_node(self, sample_size, nonzero=True):
        """
        sample node
        :param sample_size:
        :return:
        """
        if nonzero:
            return np.random.choice(self.nonzero_nodes, sample_size)
        else:
            return np.random.randint(1, self.num_node, sample_size)

    def sample_node_pair(self):
        """
        sample node pair for node feature recovery
        :param sample_size:
        :return:
        """
        sample_nodes = self.sample_node(self.sample_size*2)
        sample_pairs = sample_nodes.reshape(self.sample_size, 2)
        if self.use_superv:
            if len(self.superv['node'])>0:
                sample_superv_pairs = self.superv['node'][np.random.randint(0, self.num_superv_node, self.superv_sample_size)]
            else:
                sample_nodes = self.sample_node(self.superv_sample_size * 2)
                sample_superv_pairs = sample_nodes.reshape(self.superv_sample_size, 2)
            sample_pairs = np.concatenate([sample_pairs, sample_superv_pairs], axis=0)

        return sample_pairs

    def sample_link(self):
        """
        sample positive and negative link
        :param sample_size:
        :return:
        """
        sample_size = int((self.sample_size) / (1 + self.negative_sample_size))
        sample_link = self.links[np.random.randint(0, self.num_link, sample_size)]
        if self.use_superv:
            sample_size = int((self.superv_sample_size) / (1 + self.negative_sample_size))
            if len(self.superv['link'])>0:
                sample_superv_link = self.superv['link'][np.random.randint(0, self.num_superv_link, sample_size)]
            else:
                sample_superv_link = self.links[np.random.randint(0, self.num_link, sample_size)]
            sample_link = np.concatenate([sample_link, sample_superv_link], axis=0)
        sample_node = self.sample_node(100, False)
        sample_negative_link = []
        idx = 0
        for link in sample_link:
            for _ in range(self.negative_sample_size):
                node = sample_node[idx % 100]
                while node in self.graph[link[0]]:
                    idx += 1
                    node = sample_node[idx % 100]
                sample_negative_link.append([link[0], node])
                idx += 1

        return sample_link.tolist(), sample_negative_link

    def sample_diffusion(self):
        """
        sample node pairs from sampled diffusion
        :param sample_size:
        :return: positive pairs, negative pairs, diffusion content (torch.Float)
        """

        diff_subgraph, diff_content = random.choice(self.diffusion)
        sample_nodes = np.random.choice(diff_subgraph, self.sample_size * 2)
        sample_pairs = sample_nodes.reshape(self.sample_size, 2).tolist()

        sample_node = self.sample_node(100, False)
        sample_negative_pairs = []
        idx = 0
        for link in sample_pairs:
            for _ in range(self.negative_sample_size):
                node = sample_node[idx % 100]
                while node in diff_subgraph:
                    idx += 1
                    node = sample_node[idx % 100]
                sample_negative_pairs.append([link[0], node])
                idx += 1

        return sample_pairs, sample_negative_pairs, diff_content.repeat((self.sample_size, 1))

    def sample_diffusion_content(self):
        """
        sample node pairs from sampled diffusion
        :param sample_size:
        :return: positive pairs, negative pairs, diffusion content (torch.Float)
        """

        index = random.randint(0, self.num_diff - 1)
        diff_subgraph, diff_content = self.diffusion[index]
        sample_nodes = np.random.choice(diff_subgraph, self.sample_size * 2)
        sample_pairs = sample_nodes.reshape(self.sample_size, 2)

        if self.use_superv:
            diff_l = self.superv['diffusion'][index]
            len_l = len(diff_l)
            if len_l:
                sample_superv_pairs = diff_l[np.random.randint(0, len_l, self.superv_sample_size)]
            else:
                sample_nodes = np.random.choice(diff_subgraph, self.superv_sample_size * 2)
                sample_superv_pairs = sample_nodes.reshape(self.superv_sample_size, 2)
            sample_pairs = np.concatenate([sample_pairs, sample_superv_pairs], axis=0)

        return sample_pairs.tolist(), diff_content.repeat((len(sample_pairs), 1))

    def sample_diffusion_structure(self):
        """
        sample node pairs from sampled diffusion
        :param sample_size:
        :return: positive pairs, negative pairs, diffusion content (torch.Float)
        """

        index = random.randint(0, self.num_diff-1)
        diff_subgraph, diff_content = self.diffusion[index]

        sample_size = int((self.sample_size) / (1 + self.negative_sample_size))
        sample_nodes = np.random.choice(diff_subgraph, sample_size * 2)
        sample_pairs = sample_nodes.reshape(sample_size, 2)
        if self.use_superv:
            diff_l = self.superv['diffusion'][index]
            len_l = len(diff_l)
            sample_size = int((self.superv_sample_size) / (1 + self.negative_sample_size))
            if len_l:
                sample_superv_link = diff_l[np.random.randint(0, len_l, sample_size)]
            else:
                sample_nodes = np.random.choice(diff_subgraph, sample_size * 2)
                sample_superv_link = sample_nodes.reshape(sample_size, 2)
            sample_pairs = np.concatenate([sample_pairs, sample_superv_link], axis=0)

        sample_node = self.sample_node(100, False)
        sample_negative_pairs = []
        idx = 0
        for link in sample_pairs:
            for _ in range(self.negative_sample_size):
                node = sample_node[idx % 100]
                while node in diff_subgraph:
                    idx += 1
                    node = sample_node[idx % 100]
                sample_negative_pairs.append([link[0], node])
                idx += 1

        return sample_pairs.tolist(), sample_negative_pairs

    def sample(self, mode, return_nodes=False):
        if mode == 'node':
            samples = self.sample_node_pair()
            # samepled_features, sampled_adj, sampled_labels, nzero = self.sample_subgraph(samples, True)
            node_features = self.features.index_select(0, torch.LongTensor(samples.flatten())).view(-1, 2*self.feature_len)
            return self.sample_subgraph(samples, return_nodes=return_nodes), node_features
        elif mode == 'link':
            positive_link, negative_link = self.sample_link()
            # samepled_features, sampled_adj = self.sample_subgraph(positive_link + negative_link)
            return self.sample_subgraph(positive_link + negative_link, return_nodes=return_nodes), self.binary_label
        elif mode == 'diffusion':
            positive_pair, negative_pair, diffusion_content = self.sample_diffusion()
            # samepled_features, sampled_adj = self.sample_subgraph(positive_pair + negative_pair)
            return self.sample_subgraph(positive_pair + negative_pair, return_nodes=return_nodes), (diffusion_content, self.binary_label)
        elif mode == 'diffusion_content':
            positive_pair, diffusion_content = self.sample_diffusion_content()
            # samepled_features, sampled_adj = self.sample_subgraph(positive_pair + negative_pair)
            return self.sample_subgraph(positive_pair, return_nodes=return_nodes), diffusion_content
        elif mode == 'diffusion_structure':
            positive_pair, negative_pair = self.sample_diffusion_structure()
            # samepled_features, sampled_adj = self.sample_subgraph(positive_pair + negative_pair)
            return self.sample_subgraph(positive_pair + negative_pair, return_nodes=return_nodes), self.binary_label
        else:
            exit('unknown mode!')

