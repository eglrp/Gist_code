import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import glob as gb
import dicom
from matplotlib import cm
from skimage.measure import label, regionprops
import os

'''
去掉肌肉的淋巴结范围窗提取为 bbox坐标    
调节直径范围以 保证框住所有的淋巴结
'''


def load_extract_img(dcm_dir):
    '''
    返回 提取轮廓的窗 0~255之间
    '''
    dcm = dicom.read_file(dcm_patient[i_2])
    img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # 拿到 提取轮廓的窗，并规范化
    img_extract = (img_origin - (WL_extract - WW_extract / 2)) / WW_extract * 255  # 规范化到0-255
    # 下界0，上界255
    return np.clip(img_extract, 0, 255)

def load_node_img(dcm_dir):
    '''
    返回 提取淋巴结的窗 0~255之间
    '''
    dcm = dicom.read_file(dcm_patient[i_2])
    img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # 拿到 淋巴结范围且 仅有中间轮廓的图像
    img_node = (img_origin - (WL_node - WW_node / 2)) / WW_node * 255  # 规范化到0-255
    # 下界0，上界255
    return  np.clip(img_node, 0, 255)
def get_contours(img_input):
    '''
    得到轮廓
    '''
    temp = np.uint8(img_input)  # uint8	无符号整数（0 到 255）
    ret, img = cv2.threshold(temp, 90, 255, cv2.THRESH_BINARY)  # 二值化
    _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    return contours

def get_center_contour(contours):
    '''
    查找 最中心的轮廓
    条件：按照面积从大到小的轮廓中查找，当中心点位于该轮廓内部即为所求
    '''
    area_list = []  # 保存 每个轮廓面积
    for i_contours in contours:
        area_list.append(cv2.contourArea(i_contours))  # 计算面积
    area_index = np.argsort(-np.array(area_list))  # 面积从大到小 的下标值
    for i in range(len(area_index)):
        if 1.0 == cv2.pointPolygonTest(contours[area_index[i]], (256, 256), False):
            break  # 找到目标 contours[area_index[i]]

    # 产生仅有 中心轮廓的 mask  0-1
    max_contours_mask = np.zeros(([512,512]))
    cv2.fillConvexPoly(max_contours_mask, contours[area_index[i]], 1)  # 1 为填充值
    return max_contours_mask


#######################################################################
WL_extract, WW_extract = -360, 446  # 提取轮廓的窗宽 窗位
WL_node, WW_node = 50, 60  # 淋巴结的窗宽 窗位
dataset_path = '/home/bobo/data/NeckLymphNodes/newdataset/'  # 数据集地址

save_test_dir = '/home/bobo/data/test/test006/'

bbox_txt = 'bounding_box.txt'

expand_len = 20  # 以轮廓为中心， 每层的bbox 至少长宽为10 （将长宽不足10的 补到10x10）

# 最小外接圆直径以内的轮廓 转为 bbox  半径范围
min_radius = 0
max_radius = 20

#######################################################################

# 读取dcm
dcm_path = gb.glob(dataset_path + 'dcm/*')
dcm_path.sort()  # 排序

# 读取真值mask
mask_path = gb.glob(dataset_path + 'mask/*')
mask_path.sort()  # 排序

# 将坐标写入txt
txt_file = open(bbox_txt, 'w')
txt_file.write('index_sequence，x_min,y_min,x_max,y_max\n')

# 遍历每个病例
for i in range(len(dcm_path)):

    dcm_patient = gb.glob(dcm_path[i] + '/*.dcm')
    dcm_patient.sort()  # list [40]

    mask_patient = np.load(mask_path[i])  # [40,512,512]
    # 遍历每张图
    for i_2 in range(len(dcm_patient)):
        # 返回 提取轮廓的窗 0~255之间
        img_extract = load_extract_img(dcm_patient[i_2])

        contours = get_contours(img_extract)

        # 查找 最中心的轮廓
        max_contours_mask = get_center_contour(contours)

        # 淋巴结的窗 0~255之间
        img_node=load_node_img(dcm_patient[i_2])

        # 淋巴结范围且 仅有中间轮廓的图像
        img_result = img_node * max_contours_mask

        # plt.imsave(save_test_dir + str(i)+'_'+str(i_2) + 'result.jpg', img_result, cmap=cm.gray)  # 保存淋巴结范围且 仅有中间轮廓的图像

        # 读取对应 分割肌肉的mask
        mask_patient_single = mask_patient[i_2]
        mask_patient_single_negate = (mask_patient_single == False)  # 将0,1取反
        result = img_result * mask_patient_single_negate  # 仅有中间轮廓、且去掉肌肉的淋巴结范围的图像

        # plt.imsave(save_test_dir + str(i)+'_'+str(i_2) + 'result2.jpg', result, cmap=cm.gray)

        # 开始提取淋巴结
        ret2, img2 = cv2.threshold(np.uint8(result), 1, 1, cv2.THRESH_BINARY)  # 将result转化为uint8类型后，二值化  超过1就置为1
        _, contours2, _ = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

        save_contours = []
        # 画出所有的外切圆
        for i_contours2 in contours2:
            (x, y), radius = cv2.minEnclosingCircle(i_contours2)
            radius = int(radius)  # 半径，单位mm    int()向下取整，只取整数位
            if min_radius < radius < max_radius:  # 半径范围
                save_contours.append(i_contours2)  # 保存符合条件的轮廓

        # 产生符合条件的 mask  0-1
        result_contours_mask = np.zeros(([512,512]))

        for i_save_contours in save_contours:
            cv2.fillConvexPoly(result_contours_mask, i_save_contours, 1)  # 1 为填充值

        # plt.imsave(save_test_dir + str(i)+'_'+str(i_2) + 'result2.jpg', result_contours_mask, cmap=cm.gray) # 保留一定范围的轮廓

        # mask -> bounding box
        lablel_mask = label(result_contours_mask)  # int64->uint8   0-3 -> 0-1
        props = regionprops(lablel_mask)

        txt_file.write(str(i_2) + ',')
        for prop in props:
            # 保存 bbox坐标(左上角坐标、右下角坐标)，与 标注信息 作对比
            x_min, y_min, x_max, y_max = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]

            # # 将bbox长宽小于阈值，则bbox长宽扩大为 阈值
            # if (x_max - x_min) < expand_len:
            #     temp = (x_max + x_min) / 2
            #     x_min = temp - expand_len / 2
            #     x_max = temp + expand_len / 2
            # if (y_max - y_min) < expand_len:
            #     temp = (y_max + y_min) / 2
            #     y_min = temp - expand_len / 2
            #     y_max = temp + expand_len / 2
            x_min = x_min - expand_len / 2
            y_min = y_min - expand_len / 2
            x_max = x_max + expand_len / 2
            y_max = y_max + expand_len / 2


            txt_file.write(
                str(int(x_min)) + ',' + str(int(y_min)) + ',' + str(int(x_max)) + ',' + str(int(y_max)) + ',')

            # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
            cv2.rectangle(result, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 1)

        if not os.path.exists(save_test_dir + str(i)):
            os.makedirs(save_test_dir + str(i))
        plt.imsave(save_test_dir + str(i) + '/' + str(i_2) + '_' + 'result.jpg', result,
                   cmap=cm.gray)  # 仅有中间轮廓、且去掉肌肉的 且 过滤后存在bbox 的淋巴结范围的图像

        # props即为 所求bbox。  判断是否框住 真实淋巴结

        txt_file.write('\n')
    txt_file.write('==============EndOfCase==============')  # 一个病例结束
    txt_file.write('\n')
# 关闭txt
txt_file.close()
