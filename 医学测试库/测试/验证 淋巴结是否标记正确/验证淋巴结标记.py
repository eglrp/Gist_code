import csv
import glob as gb
import dicom
import numpy as np
import cv2
import os
'''
二维平面上框住淋巴结。验证前4条数据没问题， 所有标注淋巴结未验证

'''
# 读取csv至字典
csvFile = open("node_mark.csv", "r")
reader = csv.reader(csvFile)

WL_abdoment,WW_abdoment=60,400    # 腹窗的窗宽 窗位


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



# 加载dcm
dataset_path='/home/bobo/data/NeckLymphNodes/dataset/'  # 数据集地址
dcm_path_dicom = gb.glob(dataset_path + "dcm/*")
dcm_path_dicom.sort()  # 排序


# list_Grouping为分组存储淋巴结信息     dcm_path_dicom为分组存储dcm数据
# 开始可视化




#遍历每一个病例
for i in range(len(dcm_path_dicom)):
    mark = list_Grouping[i]
    dcm_list = gb.glob(dcm_path_dicom[i] + "/*.dcm")
    dcm_list.sort()

    #遍历每一个淋巴结
    for ii in range(len(mark)):
        x,y,longest_in_xy,begin,end=int(mark[ii][1]),int(mark[ii][2]),float(mark[ii][3]),int(mark[ii][4]),int(mark[ii][5])
        # 拿到对应的范围帧
        range_of_frames = dcm_list[begin - 1:end]    # 标记信息从1开始，而dcm下标从0开始

        # 可视化每一个淋巴结
        for iii in range(len(range_of_frames)):
            # 读取dcm
            dcm = dicom.read_file(range_of_frames[iii])
            img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            # 拿到 腹窗范围且 仅有中间轮廓的图像
            img_abdoment = (img_origin - (WL_abdoment - WW_abdoment / 2)) / WW_abdoment * 255  # 规范化到0-255
            # 下界0，上界255
            img_abdoment = np.clip(img_abdoment, 0, 255)

            x_min, x_max = int(x - longest_in_xy / 2) - 1, int(x + longest_in_xy / 2) + 1
            y_min, y_max = int(y - longest_in_xy / 2) - 1, int(y + longest_in_xy / 2) + 1
            # np_frames.append([y_min,y_max,x_min,x_max])  # 代表左上角、右下角坐标

            # # 高亮 淋巴结范围
            # img_abdoment[y_min:y_max, x_min: x_max] = 255
            # plt.imshow(img_abdoment, cmap='Greys_r')
            # plt.show()

            cv2.rectangle(img_abdoment, ( x_min,y_min), ( x_max,y_max), (255, 255, 255), 1)
            if not os.path.exists('/home/bobo/data/test/test004/'+str(i)+'/'+str(ii)):
                os.makedirs('/home/bobo/data/test/test004/'+str(i)+'/'+str(ii))
            # a/b/c   a 病例序号  b 淋巴结序号  c 该淋巴结所有图像
            cv2.imwrite('/home/bobo/data/test/test004/'+str(i)+'/'+str(ii)+'/'+str(iii)+'.jpg', img_abdoment)











