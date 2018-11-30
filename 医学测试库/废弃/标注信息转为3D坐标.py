import csv
import glob as gb
import dicom
import numpy as np
import cv2
import os
'''
标注淋巴结信息  转为3D坐标  标准（x_min,y_min,z_min,x_max,y_max,z_max)
'''
# 读取csv至字典
csvFile = open("node_mark.csv", "r")
reader = csv.reader(csvFile)

node_mark_list = []  # 保存所有淋巴结标注信息
for item in reader:
    node_mark_list.append(item)

# 按照PA分块保存
list_sorted_PA=[]
for i in range(len(node_mark_list)):
    list_sorted_PA.append(node_mark_list[i][0])

list_sorted_PA=list(set(list_sorted_PA))
list_sorted_PA.sort()

list_Grouping=[]
# 分块保存
for j in range(len(list_sorted_PA)):
    temp = []
    for k in range(len(node_mark_list)):
        if list_sorted_PA[j] ==  node_mark_list[k][0]:
            temp.append(node_mark_list[k])
    list_Grouping.append(temp)

# 此时list_Grouping为分组存储

mark_coordinate_all=[]
#遍历每一个病例
for i in range(len(list_Grouping)):
    mark = list_Grouping[i]
    mark_coordinate=[]
    #遍历每一个淋巴结
    for ii in range(len(mark)):
        x,y,longest_in_xy,begin,end=int(mark[ii][1]),int(mark[ii][2]),float(mark[ii][3]),int(mark[ii][4]),int(mark[ii][5])
        x_min, x_max = int(x - longest_in_xy / 2) - 1, int(x + longest_in_xy / 2) + 1
        y_min, y_max = int(y - longest_in_xy / 2) - 1, int(y + longest_in_xy / 2) + 1
        z_min,z_max=begin-1,end-1  # 标记信息从1开始，而dcm下标从0开始

        mark_coordinate.append((x_min,y_min,z_min,x_max,y_max,z_max))
    mark_coordinate_all.append(mark_coordinate)

# mark_coordinate_all为 分组存储的3D块的坐标
print()






















