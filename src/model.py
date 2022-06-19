import math
import torch
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
from dgl.nn import GraphConv
import torch.nn as nn
from src.CMPNN import CMPNNConv

class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, node_feature_len, hidden_dim, bond_feature_len):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = node_feature_len
        self.dim = hidden_dim

        hops = 2
        self.gnn_layers = ModuleList([])
        for i in range(self.n_layer):
            if gnn == 'cmpnn':
                self.gnn_layers.append(CMPNNConv(
                    node_feats = node_feature_len if i == 0 else self.dim,
                    edge_feats = bond_feature_len if i == 0 else self.dim,
                    out_feats = hidden_dim,
                    k = hops,
                    activation = None if i == self.n_layer - 1 else torch.relu
                ))
            elif gnn == 'gcn':
                    self.gnn_layers.append(GraphConv(in_feats=node_feature_len if i == 0 else self.dim,
                                                     out_feats=hidden_dim,
                                                     activation=None if i == n_layer - 1 else torch.relu, allow_zero_in_degree=True))
            else:
                raise ValueError('unknown GNN model')

        self.pooling_layer = SumPooling()
        self.factor = None

        #not used
        self.W_atom = nn.Linear(node_feature_len, hidden_dim)
        self.W_bond = nn.Linear(bond_feature_len, hidden_dim)
        self.lr = nn.Linear(hidden_dim*2, hidden_dim)   

    def forward(self, graph):
        h = graph.ndata['feature']
        e = graph.edata['feature']
        if self.gnn == 'cmpnn':
            for i, layer in enumerate(self.gnn_layers):
                h, e = layer(graph, h, e)
            if self.factor is None:
                self.factor = math.sqrt(self.dim) / float(torch.mean(torch.linalg.norm(h, dim=1)))
            h *= self.factor
            graph_embedding = self.pooling_layer(graph, h) 
        else:
            for i, layer in enumerate(self.gnn_layers):
                h = layer(graph, h)
            if self.factor is None:
                self.factor = math.sqrt(self.dim) / float(torch.mean(torch.linalg.norm(h, dim=1)))
            h *= self.factor
            graph_embedding = self.pooling_layer(graph, h) 
        return graph_embedding