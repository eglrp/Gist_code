import os
import glob as gb
import numpy as np
import matplotlib.pyplot as plt
import dicom
import cv2
import matplotlib.patches as patches
'''
查看是否完全框住3D块
'''
WL_abdoment,WW_abdoment=60,400    # 腹窗的窗宽 窗位
dataset_path = 'C:\\Users\\Administrator\\Desktop\\testdcm/'  # 数据集地址


dcm_path_dicom = gb.glob(dataset_path+ "*.dcm")
dcm_path_dicom.sort()  # 排序

# 标注信息 单位mm
x,y,z=185,201,96   # x,y是第96张（即最中心）dcm图的坐标
longest_in_xy=15   #  1.5cm == 15mm
num_of_frames=14   #   已消失帧序号 -  未出现帧序号

# 拿到对应的范围帧
min,max=int(z-num_of_frames/2),int(z+num_of_frames/2)
range_of_frames=dcm_path_dicom[min-1:max]

# np_frames=[] # 保存  仅有淋巴结范围的 所有帧
# 读取范围帧的dcm数组，全部保存
for i in range(len(range_of_frames)):
    # 读取dcm
    dcm = dicom.read_file(range_of_frames[i])
    img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # 拿到 腹窗范围且 仅有中间轮廓的图像
    img_abdoment = (img_origin - (WL_abdoment - WW_abdoment / 2)) / WW_abdoment * 255  # 规范化到0-255
    # 下界0，上界255
    img_abdoment = np.clip(img_abdoment, 0, 255)
    x_min, x_max = int(x-longest_in_xy/2)-1,int(x+longest_in_xy/2)+1
    y_min,y_max= int(y-longest_in_xy/2)-1,int(y+longest_in_xy/2)+1
    # np_frames.append([y_min,y_max,x_min,x_max])  # 代表左上角、右下角坐标

    # 高亮 淋巴结范围
    img_abdoment[y_min:y_max, x_min: x_max] =255
    plt.imshow(img_abdoment ,cmap='Greys_r')
    plt.show()


    # # 画矩形框 并保存查看
    # cv2.rectangle(img_abdoment, ( x_min,y_min), ( x_max,y_max), (255, 255, 255), 1)
    # cv2.imwrite('C:\\Users\\Administrator\\Desktop\\aaaa\\'+str(i)+'new.jpg', img_abdoment)




