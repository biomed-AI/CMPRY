import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src import yp_data_preprocess
from src.GNN_graphpred import GNN_graphpred
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from src.yp_data_preprocess import mol_to_dgl


def load_model(path):
    model_data = torch.load("./saved/{}.dict".format(path))
    model = GNN_graphpred(gnn='cmpnn', n_layer=2, dim=1024, num_tasks=1, node_feature_len=133, bond_feature_len=147)
    model.load_state_dict(model_data['param'])
    model.GNN_model.factor = model_data['factor']
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')
parser.add_argument('--r1', type=str, default='O=C(C1=CC=CC=C1)C2=CC=CC=C2C', help='')
parser.add_argument('--r2', type=str, default='F[B-](F)(F)C1=CC=CC=C1', help='')
parser.add_argument('--p', type=str, default='O=C(C1=CC=CC=C1)C2=CC=CC(C3=CC=CC=C3)=C2CC(O)=O', help='')

args, _ = parser.parse_known_args()




reactant = args.r1 + '.' + args.r2 + '.[K+]'
raw_graph_r = Chem.MolFromSmiles(reactant)
raw_graph_p = Chem.MolFromSmiles(args.p)
dgl_graph_r = mol_to_dgl(raw_graph_r)
dgl_graph_p = mol_to_dgl(raw_graph_p)


device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
model = load_model('net')
if torch.cuda.is_available():
    model.cuda(args.gpu)
model.eval()

react, prod = dgl_graph_r.to(device), dgl_graph_p.to(device)
pred = model(react, prod)
pred = pred.item()
pred = pred*100
print("%.2f" % pred + "%")