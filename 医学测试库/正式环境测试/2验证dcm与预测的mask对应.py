import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import glob as gb
import dicom
from matplotlib import cm
'''
得到所有序列 分割后的结果
'''
WL_extract,WW_extract=-360,446   # 提取轮廓的窗宽 窗位
WL_abdoment,WW_abdoment=60,400    # 腹窗的窗宽 窗位
dataset_path = '/home/bobo/data/NeckLymphNodes/dataset/'  # 数据集地址
predict_mask = '/home/bobo/data/test/mask_result/'    # 预测的mask地址
save_test_dir = '/home/bobo/data/test/test002/' # 保存结果

# 读取dcm
dcm_path_dicom = gb.glob(dataset_path + 'dcm/*/*.dcm')
dcm_path_dicom.sort()  # 排序

# 读取预测的mask
mask_path = gb.glob(predict_mask + '*.npy')
mask_path.sort()  # 排序

# 遍历每张图，验证dcm与mask对应
for i in range(len(dcm_path_dicom)):

    # 读取dcm
    dcm = dicom.read_file(dcm_path_dicom[i])
    img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    # 拿到 提取轮廓的窗，并规范化
    img_extract = (img_origin - (WL_extract - WW_extract / 2)) / WW_extract * 255  # 规范化到0-255
    # 下界0，上界255
    img_extract = np.clip(img_extract, 0, 255)

    dicom_np = np.uint8(img_extract)  # uint8	无符号整数（0 到 255）
    ret, img = cv2.threshold(dicom_np, 90, 255, cv2.THRESH_BINARY)  # 二值化
    im2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    # 查找 最中心的轮廓   条件：按照面积从大到小的轮廓中查找，当中心店位于该轮廓内部即为所求
    area_list = []  # 保存 每个轮廓面积
    # distance_list=[] # 保存 每个轮廓 是否包含某点
    for ii in range(len(contours)):
        area_list.append(cv2.contourArea(contours[ii]))  # 计算面积
        # # 判断图像中心点（256,256）是否位于该轮廓里面  -1代表在轮廓外面   0代表在轮廓上   1代表在轮廓内
        # distance_list.append(cv2.pointPolygonTest(contours[ii], (256, 256), False))

    area_index = np.argsort(-np.array(area_list))  # 面积从大到小 的下标值
    for iii in range(len(area_index)):
        if 1.0 == cv2.pointPolygonTest(contours[area_index[iii]], (256, 256), False):
            break  # 找到目标 contours[area_index[iii]]

    # 产生仅有 中心轮廓的 mask  0-1
    max_contours_mask = np.zeros((img_origin.shape))
    cv2.fillConvexPoly(max_contours_mask, contours[area_index[iii]], 1)  # 1 为填充值

    # 拿到 腹窗范围且 仅有中间轮廓的图像
    img_abdoment = (img_origin - (WL_abdoment - WW_abdoment / 2)) / WW_abdoment * 255  # 规范化到0-255
    # 下界0，上界255
    img_abdoment = np.clip(img_abdoment, 0, 255)

    img_result = img_abdoment * max_contours_mask
    plt.imsave(save_test_dir + str(10000 + i) + 'full_img.jpg', img_result, cmap=cm.gray)  # 保存原图

    # 读取对应mask
    mask_np = np.load(mask_path[i])
    plt.imsave(save_test_dir+str(10000+i)+'mask.jpg', mask_np,cmap = cm.gray) #保存mask

    # 仅有中间轮廓的图像与mask结合
    result =img_result * mask_np
    plt.imsave(save_test_dir+str(10000+i)+'full_mask.jpg',result,cmap = cm.gray)  #保存mask之后的原图





