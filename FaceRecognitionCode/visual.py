#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@time   : 2021/11/24 14:12
@author : hui
@file   : visual.py
@version: v0.1
@desc   : 简单说明

"""

import datetime
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image


# 展示 绘图
def display_loss(loss_list, title_str):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    index = [i for i in range(len(loss_list))]
    ax.plot(index, loss_list)  # Plot some data on the axes.
    ax.set_title(title_str)
    plt.show()


# 注释
def run():
    """
    优先显示函数内的多行注释
    """
    fv_origin = np.load('../data/Yale-processed/yale_data_gabor_fv_origin.npy')
    fv_origin = fv_origin.reshape(2414, 192, 168, 40)
    index = 5

    fig, ax = plt.subplots()


    i = 0
    pic_index = 0
    for i0, fre in enumerate([0.1, 0.6, 1, 2, 5]):
        for i1, theta in enumerate(np.linspace(0, 2 * 3.14, 8, endpoint=False)):
            if i1 % 2 == 0:

                arr = fv_origin[index, :, :, i]
                im = Image.fromarray(arr)
                im = im.convert('L')
                im.save('../data/pic/{}-F{:.2f}-T{:.2f}.png'.format(i, fre, theta))

                pic_index += 1
                plt.subplot(5, 4, pic_index)
                plt.imshow(arr, cmap=plt.cm.gray)
                plt.xticks([])
                plt.yticks([])


            i += 1
    plt.savefig('../data/pic-gabor/{}.png'.format(index))
    plt.show()
    print(fv_origin.shape)


# 主函数
if __name__ == '__main__':
    s_time = datetime.datetime.now()
    print(s_time, '程序开始运行')
    run()
    e_time = datetime.datetime.now()
    print(e_time, '运行结束，耗时', e_time - s_time)
