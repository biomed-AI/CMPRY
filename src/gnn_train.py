from glob import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from statistics import stdev
import pickle
import pandas as pd
from src.model import GNN
from src.GNN_graphpred import GNN_graphpred
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from time import time
from copy import deepcopy
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def load_model(args):
    #load pretrained model
    path = args.pretrained

    mole = GNN(gnn='cmpnn', n_layer=2, node_feature_len=133, hidden_dim=1024, bond_feature_len=147)
    model = GNN_graphpred(gnn='cmpnn', n_layer=2, dim=1024, num_tasks=1, node_feature_len=133, bond_feature_len=147)

    mole.load_state_dict(torch.load(path + '/model.pt', map_location=torch.device('cpu')))
    model.GNN_model.load_state_dict(mole.state_dict())

    return model


def eval(args, model, device, loader, save=0):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
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
    return r2, rmse


def train_model(args, model, criterion, optimizer, train_loader, valid_loader, test_loader, device):
    best_valid = -100000000

    early_stop, running_step = 2, 0
    test_step = 100
    show_step = 100
    t = time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        L = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            react, prod, y = batch
            react, prod, y = react.to(device), prod.to(device), y.to(device)
            pred = model(react, prod).reshape(-1)

            loss_mat = criterion(pred, y)
            loss = torch.sum(loss_mat) / len(y)
            L += loss
            loss.backward()
            optimizer.step()

        train_r2, train_rmse = eval(args, model, device, train_loader)
        val_r2, val_rmse = eval(args, model, device, valid_loader)

        if epoch % show_step == 1:
            test_r2, test_rmse = eval(args, model, device, test_loader, save=1)
            print('epoch: {} loss: {} lr: {}'.format(epoch, L, optimizer.param_groups[-1]["lr"]))
            print("train:\t r2 score: %.4f, rmse score: %.4f" % (train_r2, train_rmse))
            print("valid:\t r2 score: %.4f, rmse score: %.4f" % (val_r2, val_rmse))
            print("test:\t r2 score: %.4f, rmse score: %.4f" % (test_r2, test_rmse))
            print("time used: %.1fs" % (time() - t))
            print("====================================================")
            t = time()

        if epoch % test_step == 1:
            if val_r2 >= best_valid:
                best_valid = val_r2
                best_valid_test = [test_r2, test_rmse]
                running_step = 0
            else:
                running_step += 1
            
            if running_step >= early_stop:
                break
    return best_valid_test[0], best_valid_test[1]
        
def train(args, data):
    data = data[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=8)
    r2s, rmses = [], []
    for i, index in enumerate(kf.split(data)):
        train_index, test_index = index
        valid_index = np.random.choice(train_index, int(0.05*len(train_index)), replace=False)
        train_index = list(set(train_index)-set(valid_index))

        train_dataset = [data[n] for n in train_index]
        valid_dataset = [data[n] for n in valid_index]
        test_dataset = [data[n] for n in test_index]
        train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        valid_loader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print(len(train_dataset), len(test_dataset))

        model = load_model(args)
        if torch.cuda.is_available():
            model.cuda(args.gpu)
        criterion = nn.MSELoss(reduction = 'none')
        device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay, amsgrad=True)

        r_squared, rmse = train_model(args, model, criterion, optimizer, \
            train_loader, valid_loader, test_loader, device)

        print("fold {}: r2:{}  rmse:{}".format(i, r_squared, rmse))
        r2s.append(r_squared)
        rmses.append(rmse)

    r2s_avg = sum(r2s) / len(r2s)
    r2s_std = stdev(r2s)
    rmse_avg = sum(rmses) / len(rmses)
    rmse_std = stdev(rmses)

    print("average r2: %.4f, std r2: %.3f" % (r2s_avg, r2s_std))
    print("average rmse: %.4f, std rmse: %.3f" % (rmse_avg, rmse_std))


def test_external(args, data):
    train_data, external = data[0], data[1]
    train_index = list(range(len(data[0])))
    valid_index = np.random.choice(train_index, int(0.05*len(train_index)), replace=False)
    train_index = list(set(train_index) - set(valid_index)) 

    train_dataset = [train_data[n] for n in train_index]
    valid_dataset = [train_data[n] for n in valid_index]
    print('training data: {}'.format(len(train_dataset)), 'external testing data: {}'.format(len(external)))
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    ex_loader = GraphDataLoader(external, batch_size=args.batch_size, shuffle=False)

    model = load_model(args)
    if torch.cuda.is_available():
        model.cuda(args.gpu)
    criterion = nn.MSELoss(reduction = 'none')
    device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay, amsgrad=True)

    r_squared, rmse = train_model(args, model, criterion, optimizer, \
        train_loader, valid_loader, ex_loader, device)

    print("external result is r2: %.4f, rmse: %.4f" % (r_squared, rmse))