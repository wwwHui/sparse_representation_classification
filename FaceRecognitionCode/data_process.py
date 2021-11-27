#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@time   : 2021/11/27 09:08
@author : hui
@file   : data_process.py
@version: v0.1
@desc   : 简单说明

"""
import os
import random
import datetime
import numpy as np
from PIL import Image
from skimage import filters
from fishervector import FisherVectorGMM
from img_gist_feature import utils_gist


# 读取图片 特征提取
def data_process_yale(args, yale_dir='../data/CroppedYale', out_dir='../data/Yale-v4',
                      size=(16, 32), png_dir=None,
                      gabor=False, gabor_fisher_v=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    class_dir = os.listdir(yale_dir)
    length = 100
    data = np.zeros((length, size[0] * size[1]))
    if gabor:
        data_gabor = np.zeros((length, size[0] * size[1]))
        if gabor_fisher_v:
            data_gabor_fv = np.zeros((length, 192 * 168, 20))
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
            # np.save(os.path.join(out_dir, 'yale_data_gabor_fv_origin.npy'), data_gabor_fv)
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


