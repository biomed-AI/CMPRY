import os
import argparse
import yp_data_preprocess
from src import xgb_train


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5, help='the index of gpu device')
    parser.add_argument('--test', type=int, default=0, help='0: 5cv; 1: test on external')
    parser.add_argument('--dataset', type=str, default='test', help='')
    

    parser.add_argument('--lr', type=float, default=0.1, help='xgb param')
    parser.add_argument('--colsample_bytree', type=float, default=0.5, help='xgb param')
    parser.add_argument('--subsample', type=float, default=0.5, help='xgb param')
    parser.add_argument('--max_depth', type=int, default=10, help='xgb param')
    parser.add_argument('--min_child_weight', type=int, default=1, help='xgb param')

    args = parser.parse_args()
    print_setting(args)

    data = yp_data_preprocess.load_data(args)
    if args.test == 0:
        xgb_train.train(args, data)
    else:
        xgb_train.test_external(args, data)

if __name__ == '__main__':
    main()
