import argparse
import torch
import numpy as np

import random


def initialize_seeds(seedNum):
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)


def create_parser():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--model', type=str,
                        default='biirt-biased', help='type')
    parser.add_argument('--name', type=str, default='demo', help='type')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='type')
    parser.add_argument('--question_dim', type=int, default=4, help='type')
    parser.add_argument('--lr', type=float, default=1e-3, help='type')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='type')
    parser.add_argument('--inner_lr', type=float, default=1e-1, help='type')
    parser.add_argument('--inner_loop', type=int, default=5, help='type')
    parser.add_argument('--policy_lr', type=float, default=2e-3, help='type')
    parser.add_argument('--dropout', type=float, default=0.5, help='type')
    parser.add_argument('--dataset', type=str,
                        default='eedi-3', help='eedi-1 or eedi-3')
    parser.add_argument('--fold', type=int, default=5, help='type')
    parser.add_argument('--n_query', type=int, default=10, help='type')
    parser.add_argument('--seed', type=int, default=221, help='type')
    parser.add_argument('--use_cuda', default='cuda:1', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--n_qtext_vec', default=2, type=int, help='the dimension of question text')
    parser.add_argument('--ratio_0', default=0.25, type=float, help='ratio of 0/1')
    parser.add_argument('--training_type', default='iid', type=str, help='ratio of 0/1')
    parser.add_argument('--weight_ratio', default=1.0, type=float, help='weight of ratio')
    parser.add_argument('--env', default=0, type=int, help='use inv learning or not')
    parser.add_argument('--var_hyper', default=0.1, type=float, help='hyperparameter of inv learning')
    parser.add_argument('--version', default=1, type=int, help='version of dataset')
    parser.add_argument('--low_dim', default=5, type=int, help='low dim of kscd')
    parser.add_argument('--n_epoch', default=100, type=int, help='epochs')
    parser.add_argument('--wait', default=20, type=int, help='wait for early stopping')
    parser.add_argument('--reweight', default=0, type=int, help='reweight or not(1/0)')
    parser.add_argument('--lam', default=0, type=float, help='mixup lam')
    parser.add_argument('--alpha', default=0.1, type=float, help='mixup alpha')
    parser.add_argument('--mix_ratio', default=2, type=float, help='mixup ratio')
    parser.add_argument('--hyp_1', default=0.1, type=float, help='hyperparameter of mixup loss')
    # parser.add_argument('--drop_out', default=0.5, type=float, help='hyperparameter of drop_out')
    parser.add_argument('--groupDRO', default=0, type=float, help='groupDRO or not(not 0/0)')
    
    params = parser.parse_args()

    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]

    
    if params.dataset == 'eedi-3' or params.dataset == 'eedi-3_meta' or params.dataset == 'eedi-3_aug':
        params.n_question = 948
        params.train_batch_size = 512
        params.test_batch_size = 1000
        # params.n_epoch = 300
        # # params.n_epoch = 2
        # params.wait = 20
        # params.wait = 300
        params.repeat = 5
        params.question_dim = 86
    if params.dataset == 'slp_math':
        params.n_question = 222
        params.train_batch_size = 512
        params.test_batch_size = 1000
        # params.n_epoch = 500
        params.wait = 50
        params.n_epoch = 70
        if base == 'bincdm' or base =='bikancd':
            params.n_epoch = 1000
            params.wait = 250
        if base == 'biirt':
            params.n_epoch = 1000
            params.wait = 150
        # params.wait = 300
        params.repeat = 5
        params.question_dim = 31
    if params.dataset == 'eedi-1':
        params.n_question = 27613
        params.n_epoch = 750
        params.train_batch_size = 128
        params.test_batch_size = 512
        params.wait = 50
        params.repeat = 2
        params.question_dim = 388
    if params.dataset == 'assist2009':
        params.n_question = 17372
        params.train_batch_size = 50
        params.test_batch_size = 50
        if base == 'biirt':
            params.train_batch_size = 128
            params.test_batch_size = 512
        # params.seed=135
        # # params.n_epoch = 5000
        # params.n_epoch = 100
        # params.n_epoch = 2
        # params.wait = 20
        params.repeat = 5
        params.question_dim = 119
    if params.dataset == 'ednet':
        params.n_question = 13169
        params.n_epoch = 500
        params.train_batch_size = 200
        params.test_batch_size = 512
        params.wait = 25
        params.repeat = 1
    if params.dataset == 'junyi':
        params.n_question = 25785
        params.n_epoch = 750
        params.train_batch_size = 128
        params.test_batch_size = 512
        params.wait = 50
        params.repeat = 2

    return params
