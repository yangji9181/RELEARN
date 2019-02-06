import math
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GraphConvolution, MLPLayer, DecodeLink


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer=1):
        super(MLP, self).__init__()
        if layer==1:
            self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim, bias=True))
        elif layer==2:
            self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim, bias=True))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, label):
        return self.loss_fn(self.classifier(batch), label)

    def predict(self, batch, label):
        self.eval()
        _, predicted = torch.max(self.classifier(batch), 1)
        c = (predicted == label).squeeze()
        precision = torch.sum(c).item() / float(c.size(0))
        self.train()
        return predicted.cpu().numpy(), precision


class GCNDecoder(nn.Module):
    def __init__(self, device, embedding, nfeat, nhid, ncont, nrel, rel_dim, dropout, a, b, c, d, tau=1.0, hard_gumbel=False, lamb=1e-7):
        super(GCNDecoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)

        self.device = device
        self.nhid = 2*nhid
        self.nembed = nhid
        self.nrel = nrel
        self.nfeat = nfeat
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.lamb = lamb
        self.tau = tau
        self.hard_gumbel = hard_gumbel

        self.relations_mean = nn.Parameter(torch.FloatTensor(nrel, rel_dim).uniform_(-0.5 / rel_dim, 0.5 / rel_dim))
        self.relations_log_sigma = nn.Parameter(torch.FloatTensor(nrel, rel_dim).uniform_(-0.5 / rel_dim, 0.5 / rel_dim))
        self.m = torch.distributions.Normal(torch.zeros(nrel, rel_dim), torch.ones(nrel, rel_dim))

        self.encoder = GraphConvolution(nfeat, self.nembed)
        self.encoder1 = MLPLayer(2*nfeat, rel_dim)
        self.decoder = MLPLayer(self.nhid, nrel)
        self.decoder1 = MLPLayer(rel_dim, 2*nfeat, 2)  # recover node feature
        self.decoder2 = MLPLayer(rel_dim, 1, 2)    # recover graph structure
        # self.decoder2 = DecodeLink(nhid)
        self.decoder3 = MLPLayer(rel_dim, ncont, 2)    # recover diffusion content
        self.decoder4 = MLPLayer(rel_dim, 1, 2)    # recover diffusion structure
        # self.decoder4 = DecodeLink(nhid)
        self.dropout = dropout

    def forward_encoder(self, x, adj, return_pair=True):
        h = F.relu(self.encoder(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        if return_pair:
            h = h.view(-1, self.nhid)   # h_ij
        return h

    def forward_decoder(self, g_ij, nodes, prior=None):
        h_ij = self.encoder1(self.embedding(nodes).reshape(-1, self.nfeat*2))
        z_ij = F.gumbel_softmax(self.decoder(g_ij), tau=self.tau, hard=self.hard_gumbel)

        std_z = self.m.sample().to(self.device)
        rel_var = self.relations_mean + self.relations_log_sigma.exp() * std_z
        rel_var = F.dropout(rel_var, self.dropout, training=self.training)

        h_ij0 = z_ij @ rel_var

        if prior is None:
            prior = torch.ones_like(z_ij)/self.nrel

        kl = (z_ij - prior).pow(2).sum(1).mean()
        l2_loss = ((h_ij - h_ij0)**2).mean(dim=1).mean()

        return kl+l2_loss, h_ij

    def forward(self, input, adj, label, mode, nodes, prior=None):
        input = input.to(self.device)
        adj = adj.to(self.device)
        label = label.to(self.device)
        nodes = torch.LongTensor(nodes).to(self.device)
        if prior is not None: prior = prior.to(self.device)
        g_ij = self.forward_encoder(input, adj)
        decoder_loss, h_ij = self.forward_decoder(g_ij, nodes, prior)

        if mode == 'node':
            output = self.decoder1(h_ij)
            loss = self.a * (F.mse_loss(output, label) + decoder_loss)
        elif mode == 'link':
            output = self.decoder2(h_ij)
            loss = self.b * (F.binary_cross_entropy_with_logits(output, label) + decoder_loss)
        elif mode == 'diffusion_content':
            output = self.decoder3(h_ij)
            loss = self.d * (F.mse_loss(output, label) + decoder_loss)
        elif mode == 'diffusion_structure':
            output = self.decoder4(h_ij)
            loss = F.binary_cross_entropy_with_logits(output, label)
            loss = self.c * (loss + decoder_loss)
        else:
            exit('unknown mode!')

        return loss

    def generate_embedding(self, features, adj):
        features = features.to(self.device)
        adj = adj.to(self.device)
        # embedding = torch.spmm(adj, features).data.cpu().numpy()
        embedding = self.forward_encoder(features, adj, return_pair=False).data.cpu().numpy()
        return embedding

    def save_embedding(self, features, adj, path, binary=True):
        learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(self.nembed)
        learned_embed.add(list(range(len(features))), self.generate_embedding(features, adj))
        learned_embed.save_word2vec_format(fname=path, binary=binary, total_vec=len(features))

