import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dicom
# 窗宽 窗位
WL,WW=-360,446
dcm_dir='image.dcm'

dcm = dicom.read_file(dcm_dir)
img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
img_abdoment = (img_origin - (WL - WW / 2)) / WW * 255  # (x-min)/(max-min)
img_abdoment[img_abdoment < 0] = 0
img_abdoment[img_abdoment > 255] = 255
plt.subplot(1,4,1)
plt.title('ww_wl')
plt.imshow(img_abdoment, cmap='Greys_r')


#===========================================
dicom_np = np.uint8(img_abdoment)
ret, img = cv2.threshold(dicom_np,90,255, cv2.THRESH_BINARY)
im2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(dicom_np, contours, -1, (0, 255, 255), 2)
plt.subplot(1, 4, 2)
plt.title('contours')
plt.imshow(dicom_np, cmap='gray')

# 找到所有轮廓的最大面积，内部用0填充
area = []
for i in range(len(contours)):
    area.append(cv2.contourArea(contours[i])) #计算面积
max_idx = np.argmax(area)  #找到最大面积对应的轮廓下标号

# 将最大面积内部用 255填充
img_bobo=img.copy()
cv2.fillConvexPoly(img_bobo, contours[max_idx], 0)

for i in range(len(contours)):
    cv2.fillConvexPoly(img, contours[i],255)
plt.subplot(1, 4, 3)
plt.title('contours')
plt.imshow(img, cmap='gray')



plt.subplot(1, 4, 4)
plt.title('final')
plt.imshow(img-img_bobo, cmap='gray')

plt.show()