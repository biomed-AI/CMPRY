import argparse
import yp_data_preprocess
import torch
import numpy as np
from src import gnn_train as gnn_train

def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5, help='the index of gpu device')
    parser.add_argument('--test', type=int, default=0, help='0: 5-fold cross validation; 1: test on external')

    parser.add_argument('--dataset', type=str, default='test', help='')
    parser.add_argument('--pretrained', type=str, default='./pretrain/saved/cmpnn_1024', help='pretrained model path')

    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.01, help='decay')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--save', type=bool, default=False, help='save model')

    args = parser.parse_args()
    print_setting(args)
    np.random.seed(args.seed)
    data = yp_data_preprocess.load_data(args)
    if args.test == 0:
        gnn_train.train(args, data)
    else:
        gnn_train.test_external(args, data)

if __name__ == '__main__':
    main()