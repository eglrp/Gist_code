import cv2
import numpy as np
import matplotlib.pyplot as plt
aaa=cv2.imread('4mask.jpg',cv2.IMREAD_GRAYSCALE)
dicom_np = np.uint8(aaa)
ret, img = cv2.threshold(dicom_np,90,255, cv2.THRESH_BINARY)  # 二值化
im2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
plt.subplot(1, 3, 1)
plt.title('contours')
plt.imshow(im2, cmap='gray')


for i in range(4):  # 前4个轮廓填充满，若 所有轮廓填充，则会改变轮廓面积
    cv2.fillConvexPoly(img, contours[i], 255)

plt.subplot(1, 3, 2)
plt.title('contours')
plt.imshow(img, cmap='gray')

plt.show()