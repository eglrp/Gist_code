'''
测试ww   wl
'''
import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dicom

# 窗位 窗宽
WL,WW=42.5,15  #肌肉范围

# 加载dcm文件
file_dir='/home/bobo/data/NeckLymphNodes/dataset/dcm/0/'
img_path_dicom = gb.glob(file_dir+"*.dcm")
img_path_dicom.sort()  #排序


# 加载mask  [batch,w,h]
mask_np = np.load("/home/bobo/data/NeckLymphNodes/dataset/mask/0_mask_np.npy")


for i in range(len(mask_np)):


    # 读取dcm文件
    dcm = dicom.read_file(img_path_dicom[i])
    img_origin= dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    plt.subplot(1, 3, 1)
    plt.title('origin')
    plt.imshow(img_origin, cmap='Greys_r')   #显示原图


    img_abdoment = (img_origin - (WL - WW / 2)) / WW * 255  # (x-min)/(max-min)
    img_abdoment[img_abdoment < 100] = 0
    img_abdoment[img_abdoment > 150] = 255
    plt.subplot(1,3, 2)
    plt.title('ww_wl')
    plt.imshow(img_abdoment, cmap='Greys_r')  #ww wl之后的




    mask=mask_np[i]
    img_after_mask=img_abdoment* mask
    plt.subplot(1, 3, 3)
    plt.title('img_after_mask')
    plt.imshow(img_after_mask, cmap='Greys_r')   #mask之后的


    plt.show()

    # 保存
    # plt.imsave('C:\\Users\\Administrator\\Desktop\\00000\\' + str(i) + 'dcm.jpg',img_after_mask,cmap='Greys_r')


# 可视化查看
print()