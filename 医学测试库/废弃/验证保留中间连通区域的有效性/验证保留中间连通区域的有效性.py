import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dicom
# 提取轮廓的窗宽 窗位
WL,WW=-360,446
dcm_dir='aaa.dcm'

dcm = dicom.read_file(dcm_dir)
img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
img_abdoment = (img_origin - (WL - WW / 2)) / WW * 255  # (x-min)/(max-min)
img_abdoment[img_abdoment < 0] = 0
img_abdoment[img_abdoment > 255] = 255
plt.subplot(1,3,1)
plt.title('ww_wl')
plt.imshow(img_abdoment, cmap='Greys_r')


#===========================================
dicom_np = np.uint8(img_abdoment)
ret, img = cv2.threshold(dicom_np,90,255, cv2.THRESH_BINARY)  # 二值化
im2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
# cv2.drawContours(dicom_np, contours, -1, (0, 255, 255), 2)  # 填充轮廓颜色
# plt.subplot(1, 4, 2)
# plt.title('contours')
# plt.imshow(dicom_np, cmap='gray')



distance = []
for i in range(len(contours)):
    # 判断图像中心点（256,256）是否位于该轮廓里面  -1代表在轮廓外面   0代表在轮廓上   1代表在轮廓内
    distance.append(cv2.pointPolygonTest(contours[i],(250,250),False))
max_index=np.argmax(distance)  # 最大值索引

# 产生仅有 轮廓在最中间且面积较大的部分的 mask  0-1
max_contours_mask=np.zeros((img_origin.shape))
cv2.fillConvexPoly(max_contours_mask, contours[max_index], 1)


plt.subplot(1, 3, 2)
plt.title('max_contours_mask')
plt.imshow(max_contours_mask, cmap='gray')


# 拿到 腹窗范围的图像
WL2,WW2=40,350
img_abdoment2 = (img_origin - (WL2 - WW2 / 2)) / WW2 * 255  # (x-min)/(max-min)
img_abdoment2[img_abdoment2 < 0] = 0
img_abdoment2[img_abdoment2 > 255] = 255
plt.subplot(1,3,3)
plt.title('final_save_max_contours')
plt.imshow(img_abdoment2*max_contours_mask, cmap='gray')
plt.show()



aaaa=img_abdoment2*max_contours_mask



#=================================
# 加载标注信息mask
biaozhu=cv2.imread("label.png")[:,:,2]  #读取图片
biaozhu[biaozhu>0]=1

# 加载dcm

plt.subplot(1,2,1)
plt.title('origin')
plt.imshow(img_abdoment2, cmap='gray')


plt.subplot(1,2,2)
plt.title('biaozhu')
plt.imshow(biaozhu* img_abdoment2, cmap='gray')

plt.show()
bbbb=biaozhu* img_abdoment2


from sklearn import metrics
print("MSE:",metrics.mean_squared_error(aaaa, bbbb))