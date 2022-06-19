import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yp_data_preprocess
import pickle
from src.GNN_graphpred import GNN_graphpred
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_model(path):
    model_data = torch.load("./saved/{}.dict".format(path))
    model = GNN_graphpred(gnn='cmpnn', n_layer=2, dim=1024, num_tasks=1, node_feature_len=133, bond_feature_len=147)
    model.load_state_dict(model_data['param'])
    model.GNN_model.factor = model_data['factor']
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')
parser.add_argument('--dataset', type=str, default='test', help='')
parser.add_argument('--test', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')

args, _ = parser.parse_known_args()
data = yp_data_preprocess.load_data(args)
train_data, external_data = data[0], data[1]

if args.dataset == 'test':
    model = load_model('net')
elif args.dataset == 'strict_test':
    model = load_model('net_for_strict_test')
if torch.cuda.is_available():
    model.cuda(args.gpu)

device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

print('test data size:', len(external_data))

ex_loader = GraphDataLoader(external_data, batch_size=args.batch_size, shuffle=False)
model.eval()
y_true = []
y_scores = []

for step, batch in enumerate(ex_loader):
    react, prod, y = batch
    react, prod, y = react.to(device), prod.to(device), y.to(device)

    with torch.no_grad():
        pred = model(react, prod)

    y_true.append(y.view(pred.shape))
    y_scores.append(pred)


y_true = torch.cat(y_true, dim = 0).cpu().numpy()
y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
y_scores[y_scores < 0.0] = 0.0

rmse = mean_squared_error(y_true, y_scores, squared=False)
r2 = r2_score(y_true, y_scores)
mae = mean_absolute_error(y_true, y_scores)
print('r2: {}, rmse: {}, mae: {}'.format(r2, rmse, mae))