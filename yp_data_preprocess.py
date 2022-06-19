import os
import dgl
import torch
import pickle
import pysmiles
from dgl.data.utils import save_info, load_info
import pysmiles
import pandas as pd
import numpy as np
from rdkit import Chem
from src.features_mpn import bond_features, atom_features

attribute_names = ['element', 'charge', 'aromatic', 'hcount']

class YieldPredictionDataset(dgl.data.DGLDataset):
    def __init__(self, args, dname):
        self.args = args
        self.dataset = args.dataset
        self.dname = dname
        self.path = 'data/' + self.dataset + '/cache/' + self.dname
        self.reactant_graphs = []
        self.prod_graphs = []
        self.labels = []
        super().__init__(name='yield_prediction_' + dname)

    def to_gpu(self):
        if torch.cuda.is_available():
            self.labels = self.labels.to('cuda:' + str(self.args.gpu))
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.prod_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.prod_graphs]

    def save(self):
        print('saving dataset to ' + self.path + '.bin')
        save_info(self.path + '_info.pkl', {'labels': self.labels})
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.prod_graphs)

    def load(self):
        print('loading dataset from ' + self.path + '.bin')
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.prod_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]
        self.labels = load_info(self.path + '_info.pkl')['labels']
        

    def process(self):
        print('processing '  + self.dname + ' dataset')
        original_path = 'data/' + self.dataset + '/' + self.dname
        with open(original_path + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == '\n':
                    continue
                items = line.strip().split(',')
                _, reactant1, reactant2, prod, label = items[0], items[1], items[2], items[3], items[4]
                reactant = reactant1 + '.' + reactant2
                raw_graph_r = Chem.MolFromSmiles(reactant)
                raw_graph_p = Chem.MolFromSmiles(prod)
                dgl_graph_r = mol_to_dgl(raw_graph_r)
                dgl_graph_p = mol_to_dgl(raw_graph_p)
                self.reactant_graphs.append(dgl_graph_r)
                self.prod_graphs.append(dgl_graph_p)
                self.labels.append(float(label))
            self.labels = torch.Tensor(self.labels)

    def has_cache(self):
        return os.path.exists(self.path + '_reactant_graphs.bin') and os.path.exists(self.path + '_product_graphs.bin')

    def __getitem__(self, i):
        return self.reactant_graphs[i], self.prod_graphs[i], self.labels[i]

    def __len__(self):
        return len(self.reactant_graphs)

    def get_all_items(self):
        return self.reactant_graphs, self.prod_graphs, self.labels


def networkx_to_dgl(raw_graph, feature_encoder):
    # add edges
    attribute_names = ['element', 'charge', 'aromatic', 'hcount']
    src_ = [s for (s, _) in raw_graph.edges]
    dst_ = [t for (_, t) in raw_graph.edges]
    src = src_ + dst_
    dst = dst_ + src_
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    # add node features
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features
    graph.edata['feature'] = torch.randn(2*len(raw_graph.edges), 512)

    # transform to bi-directed graph with self-loops
    # graph = dgl.to_bidirected(graph, copy_ndata=True)
    # graph = dgl.add_self_loop(graph)
    return graph


def mol_to_dgl(raw_graph):
    n_node = raw_graph.GetNumAtoms()
    src, dst = [], []
    node_feat, bond_feat = [], []
    for i, atom in enumerate(raw_graph.GetAtoms()):
        node_feat.append(atom_features(atom))
    node_feat = [node_feat[i] for i in range(n_node)]

    for a1 in range(n_node):
        # src.append(a1)
        # dst.append(a1)
        # f_bonds = bond_features(None)
        # bond_feat.append(f_bonds + node_feat[a1])
        for a2 in range(a1+1, n_node):
            bond = raw_graph.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue
            f_bonds = bond_features(bond)
            src.append(a1)
            dst.append(a2)
            src.append(a2)
            dst.append(a1)
            bond_feat.append(f_bonds + node_feat[a1])
            bond_feat.append(f_bonds + node_feat[a2])

    src, dst, bond_feat = zip(*sorted(list(zip(src, dst, bond_feat)), key=lambda x:x[0]))
    graph = dgl.graph((src, dst), num_nodes=n_node)
    node_feat = torch.tensor(node_feat)
    bond_feat = torch.tensor(bond_feat)
    graph.ndata['feature'] = node_feat
    graph.edata['feature'] = bond_feat
    return graph


def load_data(args):
    data = []
    if not os.path.exists('data/'+args.dataset+'/cache/'):
        path = 'data/'+args.dataset+'/cache/'
        print('creating directory: %s' % path)
        os.mkdir(path)
    if args.test == 0:
        data.append(YieldPredictionDataset(args, 'com'))
    else:
        data.append(YieldPredictionDataset(args, 'com'))
        data.append(YieldPredictionDataset(args, 'external'))
    return data

