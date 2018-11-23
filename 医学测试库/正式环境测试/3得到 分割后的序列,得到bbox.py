import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import glob as gb
import dicom
from matplotlib import cm
from skimage.measure import label, regionprops
'''
得到所有序列 分割后的结果，并过滤后得到最终的bbox

按照 外切圆半径 来过滤
'''
WL_extract,WW_extract=-360,446   # 提取轮廓的窗宽 窗位
WL_node,WW_node=50,60    # 淋巴结的窗宽 窗位
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



    # 拿到 淋巴结范围且 仅有中间轮廓的图像
    img_node = (img_origin - (WL_node - WW_node / 2)) / WW_node * 255  # 规范化到0-255

    # 下界0，上界255
    img_node = np.clip(img_node, 0, 255)

    img_result = img_node * max_contours_mask    # 淋巴结范围且 仅有中间轮廓的图像
    mask_np = np.load(mask_path[i])   # 读取对应 分割肌肉的mask
    mask_np_negate = (mask_np == False)  # 将0,1取反
    result =img_result * mask_np_negate  # 仅有中间轮廓、且去掉肌肉的淋巴结范围的图像

    # # 可视化结果
    # plt.subplot(1, 3, 1)
    # plt.imshow(img_result,cmap='Greys_r')
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask_np,cmap='Greys_r')
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow( result,cmap='Greys_r')
    #
    # plt.show()

    # 开始提取淋巴结
    ret, img = cv2.threshold(np.uint8(result),1,1, cv2.THRESH_BINARY)  # 将result转化为uint8类型后，二值化  超过1就置为1
    im2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    # plt.title('origin')
    # plt.imshow(im2, cmap='gray')
    # plt.show()



    dst2 = np.uint8(im2.astype(int))
    im22, contours2, _ = cv2.findContours(dst2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓


    save_contours=[]
    # 画出所有的外切圆
    for iiii in range(len(contours2)):
        (x, y), radius = cv2.minEnclosingCircle(contours2[iiii])
        center = (int(x), int(y))
        radius = int(radius)  # 半径，单位mm    int()向下取整，只取整数位
        if not (radius > 40 or radius < 1):  # 半径范围  决定===================================================
            save_contours.append(contours2[iiii]) # 保存符合条件的轮廓

    # 可视化
    #         cv2.circle(dst2, center, radius, (255, 0, 0), 2)
    # cv2.imshow('a',dst2)
    # cv2.waitKey(0)


    # 产生符合条件的 mask  0-1
    result_contours_mask = np.zeros((img_origin.shape))
    for iiiii in range(len(save_contours)):
        cv2.fillConvexPoly(result_contours_mask,save_contours[iiiii], 1)  # 1 为填充值

    # # 可视化
    # plt.imshow(result_contours_mask, cmap='gray')
    # plt.show()


    # mask -> bounding box
    lablel_mask = label(result_contours_mask)  # int64->uint8   0-3 -> 0-1
    props = regionprops(lablel_mask)
    for prop in props:
        # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
        cv2.rectangle(result, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 1)

    plt.imshow(result,cmap='Greys_r')
    plt.show()

    # props即为 所求bbox。调参以 保证框住所有的淋巴结



