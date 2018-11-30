import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dicom
# 窗宽 窗位
WL,WW=40,350
dcm_dir='image.dcm'

dcm = dicom.read_file(dcm_dir)
img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
img_abdoment = (img_origin - (WL - WW / 2)) / WW * 255  # (x-min)/(max-min)
img_abdoment[img_abdoment < 0] = 0
img_abdoment[img_abdoment > 255] = 255
plt.subplot(1,3,1)
plt.title('ww_wl')
plt.imshow(img_abdoment, cmap='Greys_r')


# 加载mask  [batch,w,h]
mask_np = np.load('0_mask_np.npy')[0]
plt.subplot(1,3,2)
plt.title('mask')
plt.imshow(mask_np, cmap='Greys_r')  # mask




#===============================================================
import os
import cv2
from tqdm import tqdm
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.util.montage import montage2d as montage
img_abdoment_copy=img_abdoment.copy()
lablel_mask=label(mask_np)  # int64->uint8   0-3 -> 0-1
props = regionprops(lablel_mask)  # 求外接矩形
for prop in props:
        # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
        cv2.rectangle(img_abdoment_copy, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 5)
plt.subplot(1,3,3)
plt.title('bbox')
plt.imshow(img_abdoment_copy, cmap='Greys_r')  # mask

plt.show()