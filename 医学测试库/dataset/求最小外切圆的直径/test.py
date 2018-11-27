import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
'''
只需要移除 大块区域，即静脉动脉等    小噪声后续通过外接圆直径去除
'''

aaa=cv2.imread('1dcm.jpg',cv2.IMREAD_GRAYSCALE)
aaa=aaa[0:450,100:400]
dicom_np = np.uint8(aaa)
ret, img = cv2.threshold(dicom_np,1,1, cv2.THRESH_BINARY)  # 二值化
im2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
plt.subplot(1, 4, 1)
plt.title('all_contours')
plt.imshow(im2, cmap='gray')

#   保留大块区域

aaa=img.astype(bool) # 将0-1强转为boolean类型
# 输入需要是bool类型数据
dst=morphology.remove_small_objects(aaa,min_size=200,connectivity=1)  #dst为boolean类型
plt.subplot(1, 4, 2)
plt.title('contours_big')
plt.imshow(dst, cmap='gray')


#  原图- 大块区域 =小块区域
plt.subplot(1, 4, 3)
plt.title('origin-big')
plt.imshow(img-dst, cmap='gray')

#排除特别小的
bbb=(img-dst).astype(bool) # 将0-1强转为boolean类型
# 输入需要是bool类型数据
dst2=morphology.remove_small_objects(bbb,min_size=10,connectivity=1)  #dst为boolean类型
plt.subplot(1, 4, 4)
plt.title('result')
plt.imshow(dst2, cmap='gray')
plt.show()

dst222=dst2.astype(int)

dst222 = np.uint8(dst222)
im22, contours2, _ = cv2.findContours(dst222, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓


# # 画出所有的外切圆
# for i in range(len(contours2)):
#     (x, y), radius = cv2.minEnclosingCircle(contours2[i])
#     center = (int(x), int(y))
#     radius = int(radius)  # 半径(x,y),radius =cv2.minEnclosingCircle(contours2[0])
#     cv2.circle(dst222, center, radius, (255, 0, 0), 2)
#
#
#
# cv2.imshow('a',dst222)
# cv2.waitKey(0)
plt.subplot(1, 2,1)
plt.title('mask')
plt.imshow(dst222, cmap='Greys_r')  # mask


from skimage.measure import label, regionprops
lablel_mask=label(dst222)
props = regionprops(lablel_mask)  # 求外接矩形
for prop in props:
        # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
        cv2.rectangle(im2, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

plt.subplot(1, 2, 2)
plt.title('bounding box')
plt.imshow(im2, cmap='Greys_r')  # mask
plt.show()
print()




