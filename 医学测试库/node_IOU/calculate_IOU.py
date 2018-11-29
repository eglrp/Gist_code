# coding=utf-8
import pandas as pd
import os
import numpy as np
from node_IOU.IOU import Judge_Much_IOU

csv_dir = './node_mark.csv'
txt_dir = './bounding_box.txt'


def load_csv(csv_dir):
    """
    加载csv数据
    :param csv_dir:csv文件路径
    :return: 利用pandas读取的csv文件，dataFrame
    """
    node_mark = pd.read_csv(csv_dir)
    return node_mark


def load_txt(txt_dir):
    """
    加载txt文件数据，并且将列表按病人分割为多个
    :param txt_dir: txt文件路径
    :return: 按病人分开的列表
    """
    with open(txt_dir, 'r', encoding='UTF-8-sig') as bbox_txt:
        bbox_list = bbox_txt.readlines()
    pa_divide_list, temp = [], 0
    for i in range(len(bbox_list)):
        # 去掉最后的\n
        bbox_list[i] = bbox_list[i].rstrip('\n')
        # 按照EndOfCase来将一个列表分为多个病人列表
        if 'EndOfCase' in bbox_list[i]:
            # 将开的列表添加到一个总的列表中
            pa_divide_list.append(bbox_list[temp:i])
            temp = i + 1
    return pa_divide_list


def csv_2_bbox(x, y, length):
    """
    将csv得到的x,y,length列转化为bounding box
    :param x: pandas读取的x
    :param y: pandas读取的y
    :param length: pandas读取的length
    :return: bounding box [x_min,y_min,x_max,y_max]
    """
    x_min, y_min = x - (length / 2), y - (length / 2)
    x_max, y_max = x + (length / 2), y + (length / 2)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def bbox_str_to_bbox_list(bbox_str):
    """
    将从txt中读取到的一行bbox数据['1,222,333,333,222,123...']转化为[['222','333','333','222'],['123',...]...]
    :param bbox_str: 某病例的某一帧的bbox字符串
    :return: 去掉头尾并且按4个分开好的列表
    """
    bbox_str_list = bbox_str.split(',')[1:-1]
    bbox_str_list_np = np.array(bbox_str_list)
    # 将其分开为bbox_str_list.__len__() / 4 行，每行 4 个元素
    bbox_str_list_np = bbox_str_list_np.reshape(int(bbox_str_list.__len__() / 4), 4)
    return bbox_str_list_np.tolist()


def calculate():
    """
    计算IOU得到召回率
    :return: 无
    """
    bbox_str = load_txt(txt_dir)
    node_csv = load_csv(csv_dir)
    PA, sum, total = 1, 0, 0
    print('start calculate IOU\nnow is PA1')
    for line in range(len(node_csv)):
        if node_csv.PA[line] != PA:
            PA += 1
            print('now is PA{}'.format(PA))
        # 获取真值一行，即一个淋巴结的bbox list
        csv_bbox_list = csv_2_bbox(node_csv.x[line], node_csv.y[line], node_csv.length[line])
        # 统计所有淋巴结出现的帧数，即所有的真值
        total += (node_csv.disappear[line] - node_csv.appear[line] + 1)
        # 在对应帧，预测的bbox和真值之间的IOU
        for i in range(node_csv.appear[line] - 1, node_csv.disappear[line]):
            bbox_line_list = bbox_str_to_bbox_list(bbox_str[PA - 1][i])
            # 利用该类计算所有符合条件的IOU的累加
            sum += Judge_Much_IOU.judge_much_IOU(bbox_line_list, csv_bbox_list)
            # print(" total: ", total, " sum: ", sum, " PA: ", PA - 1, " frame: ", i)
    print('finish and the recall is {0},sum is {1},total is {2}'.format((sum / total), sum, total))


if __name__ == '__main__':
    calculate()
