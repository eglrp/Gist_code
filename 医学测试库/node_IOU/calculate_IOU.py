# coding=utf-8
import pandas as pd
import os
import numpy as np
from node_IOU.IOU import Judge_Much_IOU

csv_dir = './node_mark.csv'
txt_dir = './bounding_box.txt'


def load_csv(csv_dir):
    node_mark = pd.read_csv(csv_dir)
    return node_mark


def load_txt(txt_dir):
    with open(txt_dir, 'r', encoding='UTF-8-sig') as bbox_txt:
        bbox_list = bbox_txt.readlines()
    pa_divide_list, temp = [], 0
    for i in range(len(bbox_list)):
        bbox_list[i] = bbox_list[i].rstrip('\n')
        if 'EndOfCase' in bbox_list[i]:
            pa_divide_list.append(bbox_list[temp:i])
            temp = i + 1
    return pa_divide_list


def csv_2_bbox(x, y, length):
    x_min, y_min = x - (length / 2), y - (length / 2)
    x_max, y_max = x + (length / 2), y + (length / 2)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def bbox_str_to_bbox_list(bbox_str):
    bbox_str_list = bbox_str.split(',')[1:-1]
    bbox_str_list_np = np.array(bbox_str_list)
    bbox_str_list_np = bbox_str_list_np.reshape(int(bbox_str_list.__len__() / 4), 4)
    return bbox_str_list_np.tolist()


def calculate():
    bbox_str = load_txt(txt_dir)
    node_csv = load_csv(csv_dir)
    PA, sum, total = 1, 0, 0
    print('start calculate IOU\nnow is PA1')
    for line in range(len(node_csv)):
        if node_csv.PA[line] != PA:
            PA += 1
            print('now is PA{}'.format(PA))
        csv_bbox_list = csv_2_bbox(node_csv.x[line], node_csv.y[line], node_csv.length[line])
        total += (node_csv.disappear[line] - node_csv.appear[line]+1)
        for i in range(node_csv.appear[line] - 1, node_csv.disappear[line]):
            bbox_line_list = bbox_str_to_bbox_list(bbox_str[PA - 1][i])
            sum += Judge_Much_IOU.judge_much_IOU(bbox_line_list, csv_bbox_list)
            print(" total: ",total," sum: ",sum," PA: ",PA-1," frame: ",i)
    print('finish and the recall is {0},sum is {1},total is {2}'.format((sum / total), sum, total))


if __name__ == '__main__':
    calculate()
