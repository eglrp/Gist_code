import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dicom
plt.switch_backend('agg')

dir='/home/bobo/data/NeckLymphNodes/dataset/'

dcm_dir=dir+'dcm/'
mask_dir=dir+'mask/'

# 读取所有dcm

dcm_path_dicom = gb.glob(dcm_dir+"*/*.dcm")
dcm_path_dicom.sort()  #排序

# 读取所有mask
mask_path_dicom = gb.glob(mask_dir+"*.npy")
mask_path_dicom.sort()  #排序


mask_np = np.zeros((0,512,512))  # 以便合并，该tensor无用
for i in range(len(mask_path_dicom)):
    mask_np=np.concatenate((mask_np, np.load(mask_path_dicom[i])), axis=0)

# 将dcm和mask一一对应
dcm_mask_list=list(zip(dcm_path_dicom,mask_np))



# 验证可视化查看
dcm_dir,mask_np=dcm_mask_list[-1]
# 窗宽 窗位
WL,WW=40,350


dcm = dicom.read_file(dcm_dir)
img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
img_abdoment = (img_origin - (WL - WW / 2)) / WW * 255  # (x-min)/(max-min)
img_abdoment[img_abdoment < 0] = 0
img_abdoment[img_abdoment > 255] = 255
plt.subplot(1,2,1)
plt.title('ww_wl')
plt.imshow(img_abdoment, cmap='Greys_r')

plt.imsave('ww_wl.jpg', img_abdoment, cmap='Greys_r')


mask_np = 1 - mask_np
img_after_mask = img_abdoment * mask_np
plt.subplot(1,2,2)
plt.title('dcm_mask')
plt.imshow(img_after_mask, cmap='Greys_r')

plt.imsave('dcm_mask.jpg', img_after_mask, cmap='Greys_r')