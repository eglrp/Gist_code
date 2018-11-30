'''
将患者mask存储为numpy

cv2读取图片路径不得有中文
'''
import os
import glob as gb
import numpy as np
import cv2
import matplotlib.pyplot as plt

save_mask_dir='G:\\bbbb\\'  # 保存mask的地址

file_dir='G:\\aaaa\\'    #cv2读取图片路径不得有中文
file_path_dicom = gb.glob(file_dir+'*')
file_path_dicom.sort()  #排序




all_mask_list=[]
# 遍历所有病例
for i_file in range(len(file_path_dicom)):
    # 将一个病例的mask存储为npy
    img_path_dicom = gb.glob(file_path_dicom[i_file] + "\\*\\label.png")
    img_path_dicom.sort()  # 排序

    mask_list = []
    for i in range(len(img_path_dicom)):
        image = cv2.imread(img_path_dicom[i])

        mask = image[:, :, 2]  # mask为三通道，标记mask在第三通道上 ----------》一定可视化确认
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask)

        # 转化为0-1 mask
        mask[mask > 0] = 1
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)

        # 显示
        # plt.show()

        mask_list.append(mask)
    all_mask_list.append(mask_list)


# 遍历所有病人的mask保存
for i in range(len(all_mask_list)):
    mask_np = np.array(all_mask_list[i])
    np.save(save_mask_dir+str(i)+"_mask_np.npy", mask_np)




