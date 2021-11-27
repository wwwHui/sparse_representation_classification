#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@time   : 2021/11/27 09:06
@author : hui
@file   : main.py.py
@version: v0.5
@desc   : 简单说明

"""
import os
import sys
import random
import argparse
import datetime
import numpy as np
import pandas as pd

from data_process import data_process_yale, data_split
from face_recognition import sparse_representation_classification

# 随机数种子
def random_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)


# 参数解析
def init_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=2021,
                        help='an integer for random seed')
    parser.add_argument('--train-count', type=int, default=20,
                        help='img count for every class')
    parser.add_argument('--data-pre', action='store_true', default=False)
    parser.add_argument('--yale-dir', type=str, default='../data/CroppedYale',
                        help='yale dataset dir')
    parser.add_argument('--png-dir', type=str, default='../data/Yale-png',
                        help='dir to save png file')
    parser.add_argument('--out_dir', type=str, default='../data/Yale-v4',
                        help='dataset processed dir')
    parser.add_argument('--features', type=str, default='base',
                        help='features mode: base gabor gabor-fv')
    parser.add_argument('--only-fea', action='store_true', default=False)

    parser.add_argument('--count-list', action='store_true', default=False)

    parser.add_argument('--alm-iters', type=int, default=1000,
                        help='alm iters')
    parser.add_argument('--apg-iters', type=int, default=10,
                        help='apg-iters')

    args = parser.parse_args()

    features_list = ['base', 'gabor', 'gabor-fv', 'gist']
    if not args.features in features_list:
        print(datetime.datetime.now(), 'features error, please select from', features_list)
        sys.exit(1)

    gabor = False
    sift = False
    gabor_fv = False
    gist = False
    if args.features == 'gabor':
        gabor = True
    elif args.features == 'gabor-fv':
        gabor = True
        gabor_fv = True
    elif args.features == 'gist':
        gist = True
    parser.add_argument('--gabor', type=bool, default=gabor)
    parser.add_argument('--sift', type=bool, default=sift)
    parser.add_argument('--gabor_fv', type=bool, default=gabor_fv)
    parser.add_argument('--gist', type=bool, default=gist)

    x_hat_dir = os.path.join(args.out_dir, args.features)
    if not os.path.exists(x_hat_dir):
        os.makedirs(x_hat_dir)
        print(datetime.datetime.now(), '创建文件夹', x_hat_dir)
    x_hat_path = os.path.join(x_hat_dir, 'test_x_hat-{}.npy'.format(args.train_count))
    parser.add_argument('--x_hat_dir', type=str, default=x_hat_dir)
    parser.add_argument('--x_hat_path', type=str, default=x_hat_path)

    args = parser.parse_args()

    return args

def update_train_count(args, train_count):
    args.train_count = train_count
    args.x_hat_path = os.path.join(args.x_hat_dir, 'test_x_hat-{}.npy'.format(args.train_count))

    return args

# 初始化
def init_config():
    args = init_args()
    random_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)  # 设置小数位置为3位, 非科学计数法
    return args


# 运行函数
def run():
    args = init_config()

    # data_test()
    if args.data_pre:
        data_process_yale(args=args, yale_dir=args.yale_dir, out_dir=args.out_dir, gabor=args.gabor, gabor_fisher_v=args.gabor_fv)

    if not args.count_list:
        train_data, train_label, test_data, test_label = data_split(args.out_dir, n=args.train_count, args=args)
        sparse_representation_classification(train_data, train_label, test_data, test_label, args, example=True)

    else:  # 多进程
        start_index = 5
        max_count = 40
        count_list = [i for i in range(start_index, max_count+1, 5)]
        # count_list.reverse()

        res = {}
        for train_count in count_list:
            args = update_train_count(args, train_count)
            train_data, train_label, test_data, test_label = data_split(args.out_dir, n=train_count, args=args)
            acc = sparse_representation_classification(train_data, train_label, test_data, test_label, args)
            res[train_count] = acc

        df = pd.DataFrame.from_dict(res, orient='count', columns=['acc'])
        df.to_csv(os.path.join(args.x_hat_dir, 'acc.csv'), index=False)

        # acc_list = [[k, v] for k, v in res.items()]
        # acc_list.sort(key=lambda x: x[0])
        # print(acc_list)


# 主函数
if __name__ == '__main__':
    s_time = datetime.datetime.now()
    print(s_time, '程序开始运行')
    run()
    e_time = datetime.datetime.now()
    print(e_time, '运行结束，耗时', e_time - s_time)

