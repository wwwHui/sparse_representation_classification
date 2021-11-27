#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@time   : 2021/11/27 09:19
@author : hui
@file   : face_recognition.py
@version: v0.1
@desc   : 简单说明

"""
import random
import datetime
import numpy as np
from sklearn import preprocessing

from visual import display_loss
from algorithm import label_loss_function, augmented_lagrange_multipler


# 在稀疏表示的基础上分类
def get_category(train_label, A, y, x):
    category = np.unique(train_label)
    # print(category)
    pred = category[0]
    min_residual = label_loss_function(train_label, A, y, x, pred)
    for c in category:
        residual = label_loss_function(train_label, A, y, x, c)
        # print(c, residual, min_residual)
        if residual < min_residual:
            min_residual = residual
            pred = c

    return pred


# 系数表示分类 随机一张图片
def src_example(A, train_label, test_data, test_label, args):
    i = random.randint(0, test_data.shape[0] - 1)
    y = test_data[i, :]
    label = test_label[i]
    x_hat, loss_record = augmented_lagrange_multipler(y, A, label=label, train_label=train_label,
                                                      max_iters=args.alm_iters,
                                                      apg_iters=args.apg_iters)
    pred = get_category(train_label, A, y, x_hat)
    display_loss(loss_record['loss'], 'loss')
    display_loss(loss_record['lam'], 'lam')
    display_loss(loss_record['label_loss'], 'label_loss')
    print(datetime.datetime.now(), 'exmple, i:{}, label:{}, pred:{}'.format(i, label, pred))
    print(datetime.datetime.now(), '*' * 20, 'exmple over')


# 稀疏表示分类 所有图片 计算 acc
def sparse_representation_classification(_train_data, train_label, _test_data, test_label, args, verbose=True, example=False):
    show_interval = 1
    if not isinstance(verbose, bool):
        show_interval = verbose
        verbose = True

    scaler = preprocessing.MinMaxScaler()
    train_data = scaler.fit_transform(_train_data)
    test_data = scaler.transform(_test_data)
    length = test_data.shape[0]
    # print(test_data.shape, train_data.shape)
    A = train_data.T
    if example: src_example(A, train_label, test_data, test_label, args)
    count = 0
    err_list = []
    x_hat_arr = np.zeros((length, train_data.shape[0]))
    for i in range(length):
        y = test_data[i, :]
        label = test_label[i]
        x_hat, loss_record = augmented_lagrange_multipler(y, A,
                                                          max_iters=args.alm_iters,
                                                          apg_iters=args.apg_iters)
        pred = get_category(train_label, A, y, x_hat)

        x_hat_arr[i] = x_hat
        if pred == label:
            count += 1
        else:
            err_list.append(i)
        if verbose and i % show_interval == 0:
            print('\r{:0>4d}/{:0>4d}/{:0>4d}, acc:{:.2f}%, {}:{}-{}    '.
                  format(count, i + 1, length, 100 * count / (i + 1), pred == label, label, pred), end='')
    if verbose:
        print('\r', end='')
    print(datetime.datetime.now(),
          '{:0>4d}/{:0>4d}, acc:{:.2f}%'.format(count, length, 100 * count / length))
    # print(datetime.datetime.now(), '分类错误的图片序号是：', err_list)
    np.save(args.x_hat_path, x_hat_arr)
    print(datetime.datetime.now(), 'saved file', args.x_hat_path)
    return 100 * count / length


