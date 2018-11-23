'''
加载dcm和mask，查看是否一致
'''
import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dicom

# 窗宽 窗位
WL,WW=40,350

# 加载dcm文件
file_dir='G:\\0\\'
img_path_dicom = gb.glob(file_dir+"*.dcm")
img_path_dicom.sort()  #排序


# 加载mask  [batch,w,h]
mask_np = np.load("G:\\0_mask_np.npy")


for i in range(len(mask_np)):
    # 读取dcm文件
    dcm = dicom.read_file(img_path_dicom[i])
    img_origin= dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # plt.subplot(1, 3, 1)
    # plt.title('origin')
    # plt.imshow(img_origin, cmap='Greys_r')


    img_abdoment = (img_origin - (WL - WW / 2)) / WW * 255  # (x-min)/(max-min)
    img_abdoment[img_abdoment < 100] = 0
    img_abdoment[img_abdoment > 150] = 255
    # plt.subplot(1,3, 2)
    # plt.title('ww_wl')
    # plt.imshow(img_abdoment, cmap='Greys_r')

    mask=mask_np[i]
    mask= 1-mask
    img_after_mask=img_abdoment* mask

    # img_after_mask[img_after_mask>180]=255
    #
    # img_after_mask[img_after_mask < 80] = 0
    # plt.subplot(1, 3, 3)
    # plt.title('img_after_mask')
    # plt.imshow(img_after_mask, cmap='Greys_r')


    # plt.show()

    # 保存
    # plt.imsave('C:\\Users\\Administrator\\Desktop\\00000\\' + str(i) + 'dcm.jpg',img_after_mask,cmap='Greys_r')


# 可视化查看
print()