# coding=utf-8
import os
import numpy as np
from IOU import Judge_Much_IOU

truth_dir = './truth_bounding_box.txt'
bb_dir = './bounding_box.txt'

def load_txt(txt_dir):
    """
    加载txt文件数据，并且将列表按病人分割为多个
    :param txt_dir: txt文件路径
    :return: 按病人分开的列表
    """
    with open(txt_dir, 'r', encoding='UTF-8-sig') as bbox_txt:
        bbox_list = bbox_txt.readlines()
    bbox_list.pop(0)
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

def bbox_str_to_bbox_list(bbox_str,flag=0):
    """
    将从txt中读取到的一行bbox数据['1,222,333,333,222,123...']转化为[['222','333','333','222'],['123',...]...]
    :param bbox_str: 某病例的某一帧的bbox字符串
    :return: 去掉头尾并且按4个分开好的列表
    """
    bbox_str_list = bbox_str.split(',')[1:-1]
    bbox_str_list_np = np.array(bbox_str_list)
    # 将其分开为bbox_str_list.__len__() / 4 行，每行 4 个元素
    bbox_str_list_np = bbox_str_list_np.reshape(int(bbox_str_list.__len__() / 4), 4)
    if flag ==1:
        bbox_str_list_frame = int(bbox_str.split(',')[0])
        return {'line':bbox_str_list_frame,'list':bbox_str_list_np.tolist()}
    return bbox_str_list_np.tolist()


def calculate():
    """
    计算IOU得到召回率
    :return: 无
    """
    bbox_str = load_txt(bb_dir)
    true_str = load_txt(truth_dir)
    sum, total = 0, 0
    print('start...')
    for PA in range(len(true_str)):
        # 获取真值一行，即一个淋巴结的bbox list
        print('now is PA{}...'.format(PA))
        for line in range(len(true_str[PA])):
            true_dict = bbox_str_to_bbox_list(true_str[PA][line],1)
            # 统计所有淋巴结出现的帧数，即所有的真值
            total += len(true_dict['list'])
            # 在对应帧，预测的bbox和真值之间的IOU
            for i in range(len(true_dict['list'])):
                bbox_line_list = bbox_str_to_bbox_list(bbox_str[PA][true_dict['line']])
                #print(bbox_line_list,'bboxlist')
                # 利用该类计算所有符合条件的IOU的累加
                sum += Judge_Much_IOU.judge_much_IOU(bbox_line_list, true_dict['list'][i])
                #print(" total: ", total, " sum: ", sum, " PA: ", PA + 1, " frame: ", true_dict['line'])
    print('finish and the recall is {0},sum is {1},total is {2}'.format((sum / total), sum, total))


if __name__ == '__main__':
    calculate()
