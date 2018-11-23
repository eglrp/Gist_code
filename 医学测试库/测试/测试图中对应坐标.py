
import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dicom

WL_abdoment,WW_abdoment=60,400    # 腹窗的窗宽 窗位

# 读取dcm
dcm = dicom.read_file('bobo.dcm')
img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

# 拿到 腹窗范围且 仅有中间轮廓的图像
img_abdoment = (img_origin - (WL_abdoment - WW_abdoment / 2)) / WW_abdoment * 255  # 规范化到0-255
img_abdoment = np.clip(img_abdoment, 0, 255) # 下界0，上界255

plt.subplot(1, 2, 1)
plt.imshow( img_abdoment,cmap='Greys_r')


img_abdoment[0:200,0:100]=255
plt.subplot(1, 2, 2)
plt.imshow( img_abdoment,cmap='Greys_r')


plt.show()
print()
