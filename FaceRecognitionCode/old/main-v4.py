#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Time   : 2021/11/09 16:59
@author : hui
@file   : main.py
@desc   : 用稀疏表示实现人脸识别
"""

import os
import random
import sys
import argparse
import datetime
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import preprocessing
from skimage import filters, io, color
from fishervector import FisherVectorGMM

import matplotlib.pyplot as plt


from img_gist_feature import utils_gist


# 读取图片 特征提取
def data_process_yale(args, yale_dir='../data/CroppedYale', out_dir='../data/Yale-v4',
                      size=(16, 32), png_dir=None,
                      gabor=False, sift=False, gabor_fisher_v=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    class_dir = os.listdir(yale_dir)
    length = 100
    data = np.zeros((length, size[0] * size[1]))
    if gabor:
        data_gabor = np.zeros((length, size[0] * size[1]))
        if gabor_fisher_v:
            data_gabor_fv = np.zeros((length, 192 * 168, 20))
    if sift:
        data_sift = np.zeros((length, size[0] * size[1]))
    if args.gist:
        data_gist = np.zeros((length, 512))
        gist_helper = utils_gist.GistUtils()
    labels = []
    count = 0
    
    for dir in class_dir:
        label = int(dir[-2:])
        if png_dir is not None:
            temp_out_dir = os.path.join(png_dir, dir)
            if not os.path.exists(temp_out_dir):
                os.makedirs(temp_out_dir)
        for file in os.listdir(os.path.join(yale_dir, dir)):
            if file.endswith('pgm') and 'Ambient' not in file:
                print('\r正在处理第{}张图片'.format(count), end='')
                filepath = os.path.join(yale_dir, dir, file)
                im = Image.open(filepath)  # 读取文件
                img_arr = np.array(im)
                img_arr_small = np.array(im.resize(size))
                if png_dir is not None:
                    outpath = os.path.join(png_dir, dir, file.replace('pgm', 'png'))
                    im.save(outpath)
                img = img_arr_small.flatten()
                if gabor:
                    real, imag = filters.gabor(img_arr, frequency=0.6, theta=0, n_stds=5)
                    img_mod = np.sqrt(real.astype(float) ** 2 + imag.astype(float) ** 2)  # 取模
                    img_gabor = img_mod.copy()
                    img_gabor.resize(size)
                    # print(img_gabor.shape)
                    img_gabor = img_gabor.flatten()
                    if gabor_fisher_v:
                        img_gabor_fv = np.zeros((192 * 168, 20))
                        i = 0
                        for fre in [0.1, 0.6, 1, 2, 5]:
                            for theta in np.linspace(0, 2 * 3.14, 4, endpoint=False):
                                real, imag = filters.gabor(img_arr, frequency=fre, theta=theta, n_stds=5)
                                img_mod = np.sqrt(real.astype(float) ** 2 + imag.astype(float) ** 2)  # 取模
                                img_gabor_fv[:, i] = img_mod.copy().flatten()
                                i += 1

                    # print(np.array(im).shape, real.shape, img_mod.shape, img_gabor.shape)
                if sift:
                    pass
                    # sift2d = cv2.xfeatures2d.SIFT_create()
                    # kp, des = sift2d.detectAndCompute(img_arr, None)
                    # print('kp', kp)
                    # print(filepath, end=' ')
                    # print(des.shape)
                if args.gist:
                    img_gist = gist_helper.get_gist_vec(img_arr, mode="gray")
                    # print("shape ", np_gist.shape)
                if count == length:
                    data = np.concatenate((data, data), axis=0)
                    length *= 2
                    if gabor:
                        data_gabor = np.concatenate((data_gabor, data_gabor), axis=0)
                        if gabor_fisher_v:
                            data_gabor_fv = np.concatenate((data_gabor_fv, data_gabor_fv), axis=0)
                    if args.gist:
                        data_gist = np.concatenate((data_gist, data_gist), axis=0)
                data[count, :] = img
                if gabor:
                    data_gabor[count, :] = img_gabor
                    if gabor_fisher_v:
                        data_gabor_fv[count, :] = img_gabor_fv
                if args.gist:
                    data_gist[count, :] = img_gist
                labels.append(label)
                count += 1
    print('\r', datetime.datetime.now(), ' 图片特征提取完成，共{}张图片'.format(count), sep='')
    data = data[:count, :]
    labels = np.array(labels, dtype='int8')
    print(datetime.datetime.now(), 'data shape:', data.shape, ', label count:', len(labels))
    np.save(os.path.join(out_dir, 'yale_data.npy'), data)
    np.save(os.path.join(out_dir, 'yale_labels.npy'), labels)
    print(datetime.datetime.now(), '基础特征保存完毕')

    if gabor:
        data_gabor = data_gabor[:count, :]
        print(datetime.datetime.now(), 'data gabor shape:', data_gabor.shape)
        np.save(os.path.join(out_dir, 'yale_data_gabor.npy'), data_gabor)
        print(datetime.datetime.now(), 'Gabor特征保存完毕', )
        if gabor_fisher_v:
            print(datetime.datetime.now(), FisherVectorGMM, end=' ')
            data_gabor_fv = data_gabor_fv[:count, :]
            np.save(os.path.join(out_dir, 'yale_data_gabor_fv_origin.npy'), data_gabor_fv)
            fv_gmm = FisherVectorGMM(n_kernels=3).fit(data_gabor_fv)
            fisher_v = fv_gmm.predict(data_gabor_fv)
            fisher_v = fisher_v.reshape((count, 120))
            print(datetime.datetime.now(), 'data gabor shape:', fisher_v.shape)
            np.save(os.path.join(out_dir, 'yale_data_gabor_fv.npy'), fisher_v)
            print(datetime.datetime.now(), 'Gabor-FV特征保存完毕', )
    if args.gist:
        data_gist = data_gist[:count, :]
        np.save(os.path.join(out_dir, 'yale_data_gist.npy'), data_gist)
    return data, labels


# 数据集划分
def data_split(data_dir, args, n=3, verbose=True):
    data_path = os.path.join(data_dir, 'yale_data.npy')
    labels_path = os.path.join(data_dir, 'yale_labels.npy')
    data = np.load(data_path)
    labels = np.load(labels_path)
    print(datetime.datetime.now(), 'loaded base data and label')
    if args.gabor:
        if args.gabor_fv:
            data_gabor = np.load(os.path.join(data_dir, 'yale_data_gabor_fv.npy'))
            print(datetime.datetime.now(), 'loaded gabor fv data')
        else:
            data_gabor = np.load(os.path.join(data_dir, 'yale_data_gabor.npy'))
            print(datetime.datetime.now(), 'loaded gabor data')
        if args.only_fea:
            data = data_gabor
        else:
            data = np.append(data, data_gabor, axis=1)

    elif args.gist:
        data_gist = np.load(os.path.join(data_dir, 'yale_data_gist.npy'))
        print(datetime.datetime.now(), 'loaded gist data')
        if args.only_fea:
            data = data_gist
        else:
            data = np.append(data, data_gist, axis=1)
    train_index_list = []

    category = np.unique(labels)
    for c in category:
        c_list = [i for i, a in enumerate(labels) if c == a]
        train_list = random.sample(c_list, n)
        train_index_list.extend(train_list)

    test_index_list = [i for i in range(len(labels)) if i not in train_index_list]
    train_data = data[train_index_list, :]
    train_label = labels[train_index_list]
    test_data = data[test_index_list, :]
    test_label = labels[test_index_list]
    if verbose:
        print(datetime.datetime.now(), 'train count', n,
              'train data shape', train_data.shape, 'test data shape', test_data.shape)

    return train_data, train_label, test_data, test_label


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
def apg_lagrange(x0, A, y, mu, lam, L, max_iters=10, loss_stop=False):
    # (m, n) = A.shape
    x = x0
    x_old = x
    t = 1
    t_old = t
    i = 1
    loss = loss_function(x, y, A, lam, mu)
    loss_old = loss
    while i < max_iters:
        # print(i, x)
        t = (1 + np.sqrt(1 + 4 * t_old * t_old)) / 2
        beta = (t_old - 1) / t
        p = x + beta * (x - x_old)
        Ap = np.matmul(A, p)
        Ap_y = np.matmul(A, p) - y
        w = p - (mu * np.matmul(A.T, Ap_y) + np.matmul(A.T, lam)) / (L * mu)
        x_old = x
        x = soft(w, 1 / (L * mu))
        loss = loss_function(x, y, A, lam, mu)
        if loss_stop and loss > loss_old:
            x = x_old
            break
        loss_old = loss

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
        x = apg_lagrange(x0, A, y, mu, lam, L, max_iters=apg_iters, loss_stop=loss_stop)
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


# 展示 绘图
def display_loss(loss_list, title_str):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    index = [i for i in range(len(loss_list))]
    ax.plot(index, loss_list)  # Plot some data on the axes.
    ax.set_title(title_str)
    plt.show()


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


# 测试用 无实际用处
def data_test():
    # 准备数据
    n = 100
    r = 0.2
    m = int(0.5 * n)
    k = int(r * n)
    A = np.random.random((m, n))
    x = np.random.random(n)
    index_list = np.random.choice(n, n - k).tolist()
    x[index_list] = 0
    y = np.matmul(A, x)
    print(x)
    x_hat, loss_record = augmented_lagrange_multipler(y, A, origin_x=x)
    print(x_hat)
    # print(loss_list)

    # display_loss(loss_record['loss'])
    # display_loss(loss_record['err'])
    print('error:', loss_record['err'][-1])

    print('data_test function over')


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
