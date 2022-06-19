from email import message
import math
import torch
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, WeightAndSum
from torch.nn import ModuleList
from torch.nn.functional import one_hot, normalize, logsigmoid
import torch.nn as nn
import numpy as np
import pandas as pd
from CMPNNConv import CMPNNConv

class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, feature_len, hidden_dim, bond_feature_len=147):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = hidden_dim

        self.gnn_layers = ModuleList([])

        for i in range(self.n_layer):
            if gnn == 'cmpnn':
                self.gnn_layers.append(CMPNNConv(
                    node_feats = feature_len if i == 0 else self.dim,
                    edge_feats = bond_feature_len if i == 0 else self.dim,
                    out_feats = hidden_dim,
                    k = 2,
                    activation = None if i == self.n_layer - 1 else torch.relu
                ))

        self.pooling_layer = SumPooling()
        self.factor = None
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

        self.W_atom = nn.Linear(feature_len, hidden_dim)
        self.W_bond = nn.Linear(bond_feature_len, hidden_dim)
        self.lr = nn.Linear(hidden_dim*2, hidden_dim)   

    def forward(self, graph):
        h = graph.ndata['feature']
        e = graph.edata['feature']
        for layer in self.gnn_layers:
            h, e = layer(graph, h, e)
        if self.factor is None:
            self.factor = math.sqrt(self.dim) / float(torch.mean(torch.linalg.norm(h, dim=1)))
        h *= self.factor
        graph_embedding = self.pooling_layer(graph, h) #2048(batch_size)*embedding_dim
        return graph_embedding