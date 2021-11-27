#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@time   : 2021/11/27 09:09
@author : hui
@file   : algorithm.py
@version: v0.1
@desc   : 简单说明

"""

import datetime
import numpy as np


# 损失函数
def loss_function(x, y, A, lam, mu):
    y_hat = np.matmul(A, x)
    err = y_hat - y
    err_norm = np.linalg.norm(err, 2)
    loss = np.linalg.norm(x, 1) + np.matmul(lam, err) + mu * err_norm * err_norm / 2
    # print('***', np.sum(x), np.matmul(lam, err), err_norm, loss)
    return loss


# 误差计算 函数
def label_loss_function(train_label, A, y, x, label):
    index = train_label == label
    index = ~index
    # print(index)
    x0 = x.copy()
    x0[index] = 0
    loss = np.linalg.norm(np.matmul(A, x0) - y, 2)
    return loss


# soft 函数
def soft(w, lam):
    b = np.abs(w) - lam
    b[b < 0] = 0
    soft_thresh = np.sign(w) * b
    return soft_thresh

# apg 函数 被 alm 调用
def apg_lagrange(x0, A, y, mu, lam, L, max_iters=10):
    # (m, n) = A.shape
    x = x0
    x_old = x
    t = 1
    t_old = t
    i = 1
    while i < max_iters:
        # print(i, x)
        t = (1 + np.sqrt(1 + 4 * t_old * t_old)) / 2
        beta = (t_old - 1) / t
        p = x + beta * (x - x_old)
        Ap_y = np.matmul(A, p) - y
        w = p - (mu * np.matmul(A.T, Ap_y) + np.matmul(A.T, lam)) / (L * mu)
        x_old = x
        x = soft(w, 1 / (L * mu))
        t_old = t
        i += 1

    return x

# alm 增广拉格朗日 函数
def augmented_lagrange_multipler(y, A, max_iters=1000, loss_stop=False,
                                 origin_x=None, label=None, train_label=None,
                                 beta=2, mu=0.5, apg_iters=0):
    V, D = np.linalg.eig(np.matmul(A.T, A))  # eig函数有两个返回值，D为特征向量，V为特征值
    L = max(V).real
    mu_max = 1
    (m, n) = A.shape
    x0 = np.random.rand(n).T
    lam = np.ones(m).T

    k = 1
    loss = loss_function(x0, y, A, lam, mu)
    loss_old = loss
    loss_record = {'loss': [loss], 'lam':[]}
    if origin_x is not None:
        loss_record['err'] = [np.linalg.norm(x0 - origin_x, 2)]
    if label is not None and train_label is not None:
        loss_record['label_loss'] = [label_loss_function(train_label, A, y, x0, label)]
    while k < max_iters:
        x = apg_lagrange(x0, A, y, mu, lam, L, max_iters=apg_iters)
        loss = loss_function(x, y, A, lam, mu)
        if loss_stop and loss > loss_old:
            x = x0
            break
        loss_old = loss
        loss_record['loss'].append(loss)
        loss_record['lam'].append(np.mean(np.abs(lam)))
        lam = lam + mu * (np.matmul(A, x) - y)
        mu = min(beta * mu, mu_max)
        x0 = x
        k += 1
        if origin_x is not None:
            loss_record['err'].append(np.linalg.norm(x - origin_x, 2))
        if label is not None and train_label is not None:
            loss_record['label_loss'].append(label_loss_function(train_label, A, y, x0, label))

    return x, loss_record


# 测试用
def data_test():
    # 准备数据
    n = 100  # 字典样本数量
    r = 0.2
    m = int(0.5 * n)  # 样本特征数
    k = int(r * n)
    A = np.random.random((m, n))   # 字典
    x = np.random.random(n)  # 系数
    index_list = np.random.choice(n, n - k).tolist()
    x[index_list] = 0
    y = np.matmul(A, x)  # 待测样本
    print(x)
    # alm 估计
    x_hat, loss_record = augmented_lagrange_multipler(y, A, origin_x=x)
    print(x_hat)  # 估计系数
    # print(loss_list)

    # display_loss(loss_record['loss'])
    # display_loss(loss_record['err'])
    print('error:', loss_record['err'][-1])

    print('data_test function over')


# 注释
def run():
    """
    优先显示函数内的多行注释
    """
    data_test()


# 主函数
if __name__ == '__main__':
    s_time = datetime.datetime.now()
    print(s_time, '程序开始运行')
    run()
    e_time = datetime.datetime.now()
    print(e_time, '运行结束，耗时', e_time - s_time)
