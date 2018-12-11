import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import glob as gb
import dicom
from matplotlib import cm
from skimage.measure import label, regionprops
import os
from utils.config import opt
import xml.etree.ElementTree as ET


class getRecall():
    '''
    计算二维召回率
    '''

    def __init__(self, result_test):
        self.result_test = np.array(result_test)  # 预测的mask[batch,512,512]

    def getResult(self):
        bbox_predict = self.getPredictBbox()
        bbox_truth = self.getTruthBbox()

        np.save('bbox_predict.npy', np.array(bbox_predict))
        np.save('bbox_truth.npy', np.array(bbox_truth))
        return self.calculateRecall(bbox_predict, bbox_truth)

    def getTruthBbox(self):
        xml_path = gb.glob(opt.xml_dir + "*")
        xml_path.sort()  # 排序

        return self.get_bbox(xml_path)

    def calculateRecall(self, bbox_predict, bbox_truth):
        '''
        实时计算recall
        :param bbox_predict: 预测bbox
        :param bbox_truth: 真实值bbox
        :return: 召回率
        '''
        sum, total = 0, 0
        for PA in range(len(bbox_predict)):
            for i in range(len(bbox_truth[PA])):
                true_dict = self.list_to_bbox(bbox_truth[PA][i], 1)
                total += len(true_dict['list'])
                for j in range(len(true_dict['list'])):
                    bbox_line_list = self.list_to_bbox(bbox_predict[PA][true_dict['line']])
                    # 利用该类计算所有符合条件的IOU的累加
                    sum += self.judge_much_IOU(bbox_line_list, true_dict['list'][j])
        return (sum / total + 0.00001)

    def calcIOU(self, one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h):
        """
        计算IOU
        :param one_x: 预测值bbox 中心坐标 x
        :param one_y: 预测值bbox 中心坐标 y
        :param one_w: 预测值bbox 宽w
        :param one_h: 预测值bbox 高h
        :param two_x: 真值bbox 中心坐标 x
        :param two_y: 真值bbox 中心坐标 y
        :param two_w: 真值bbox 宽w
        :param two_h: 真值bbox 高h
        :return: IOU值
        """
        # 判断是否相交
        if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
            # 计算IOU
            lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
            lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))
            rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
            rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))
            inter_w = abs(rd_x_inter - lu_x_inter)
            inter_h = abs(lu_y_inter - rd_y_inter)
            inter_square = inter_w * inter_h
            union_square = (one_w * one_h) + (two_w * two_h) - inter_square
            calcIOU = inter_square / (two_w * two_h)
        else:
            calcIOU = 0
        return calcIOU

    def judge_much_IOU(self, box_list, this_box_list):
        """
        判断IOU是否匹配
        :param box_list: 预测值一帧的多个bbox eg: box_list[[1,2,3,4],[1,2,3,4]]
        :param this_box_list: 真值bbox eg: this_box_list [xmin,ymin,xmax,ymax]
        :return: 匹配返回1，不匹配返回0
        """
        xmin, ymin, xmax, ymax = this_box_list
        # 计算真值bbox中心及宽高
        two_x, two_y, two_w, two_h = xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2, xmax - xmin, ymax - ymin
        for onelist in box_list:
            xmin_, ymin_, xmax_, ymax_ = onelist
            # 计算预测值bbox中心及宽高
            one_x, one_y, one_w, one_h = xmin_ + (xmax_ - xmin_) / 2, ymin_ + (
                    ymax_ - ymin_) / 2, xmax_ - xmin_, ymax_ - ymin_
            # 计算IOU
            result = self.calcIOU(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h)
            if result > opt.iou_threshold:
                # 这里设置阈值，现在为若有相交，则返回1
                return 1
        return 0  # 表示该帧没有与真值相交的bbox

    def list_to_bbox(self, blist, flag=0):
        """
        将一行bbox 的 list数据变为 4个一组的bbox数据
        :param bbox_str: 某病例的某一帧的bbox字符串
        :return: 去掉头尾并且按4个分开好的列表
        """
        blist_np = np.array(blist[1:])
        # 将其分开为bbox_str_list.__len__() / 4 行，每行 4 个元素
        blist_np = blist_np.reshape(int(blist.__len__() / 4), 4)
        if flag == 1:
            blist_frame = blist[0]
            return {'line': blist_frame, 'list': blist_np.tolist()}
        return blist_np.tolist()

    def parse_rec(self, index, filename):
        """
        解析一个 PASCAL VOC xml file
        """
        tree = ET.parse(filename)
        # 存储一张图片中的所有物体
        objects = []
        objects.append(index)
        # 遍历一张图中的所有物体
        for obj in tree.findall('object'):
            bbox = obj.find('bndbox')
            # 从原图左上角开始为原点，向右为x轴，向下为y轴。左上角（xmin，ymin）和右下角(xmax,ymax)
            objects.append(int(float(bbox.find('xmin').text)))
            objects.append(int(float(bbox.find('ymin').text)))
            objects.append(int(float(bbox.find('xmax').text)))
            objects.append(int(float(bbox.find('ymax').text)))

        return objects

    def get_bbox_coordinate(self, i_single_xml_path):
        '''
        得到xml里所有bbox坐标
        '''
        index = int(i_single_xml_path.replace('.xml', '')[-5:]) - 1  # xml文件名序号从1开始，故-1 变为从0开始
        single_bbox_info = self.parse_rec(index, i_single_xml_path)
        return single_bbox_info

    def get_single_bbox(self, single_xml_path):
        '''
        通过 一个序列的地址 得到 该序列的bbox
        '''
        single_xml_path = gb.glob(single_xml_path + "/*.xml")
        single_xml_path.sort()  # 排序

        # 遍历所有xml文件
        all_bbox_info = []
        for i_single_xml_path in single_xml_path:
            all_bbox_info.append(self.get_bbox_coordinate(i_single_xml_path))
        return all_bbox_info

    def write_txt(self, bbox_txt_dir, result):
        '''
        将结果写入txt文件中
        '''
        # 将坐标写入txt
        txt_file = open(bbox_txt_dir, 'w')
        txt_file.write('index_sequence，x_min,y_min,x_max,y_max\n')
        # 遍历每个序列
        for single_result in result:
            for info in single_result:
                for single_info in info:
                    txt_file.write(str(single_info) + ',')
                txt_file.write('\n')
            txt_file.write('==============EndOfCase==============')  # 一个序列结束
            txt_file.write('\n')
        txt_file.write('\n')

    def get_bbox(self, xml_path):
        bbox_list = []
        # 遍历每个序列
        for single_xml_path in xml_path:
            bbox_list.append(self.get_single_bbox(single_xml_path))
        return bbox_list

    def getPredictBbox(self):
        # 读取dcm
        dcm_path = gb.glob(opt.dataset_path + 'dcm/*')
        dcm_path.sort()  # 排序

        result_bbox = []

        mask_index = 0
        # 遍历每个病例
        for i in range(len(dcm_path)):

            dcm_patient = gb.glob(dcm_path[i] + '/*.dcm')
            dcm_patient.sort()  # list [40]

            result_case = []

            # 遍历每张图
            for i_2 in range(len(dcm_patient)):

                result_dcm = []

                # 返回 提取轮廓的窗 0~255之间
                img_extract = self.load_extract_img(dcm_patient[i_2])

                contours = self.get_contours(img_extract)

                # 查找 最中心的轮廓
                max_contours_mask = self.get_center_contour(contours)

                # 淋巴结的窗 0~255之间
                img_node = self.load_node_img(dcm_patient[i_2])

                # 淋巴结范围且 仅有中间轮廓的图像
                img_result = img_node * max_contours_mask

                # 读取对应 分割肌肉的mask
                mask_patient_single = self.result_test[mask_index]
                mask_index = mask_index + 1

                mask_patient_single_negate = (mask_patient_single == False)  # 将0,1取反
                result = img_result * mask_patient_single_negate  # 仅有中间轮廓、且去掉肌肉的淋巴结范围的图像

                # 开始提取淋巴结
                ret2, img2 = cv2.threshold(np.uint8(result), 1, 1, cv2.THRESH_BINARY)  # 将result转化为uint8类型后，二值化  超过1就置为1
                _, contours2, _ = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

                save_contours = []
                # 画出所有的外切圆
                for i_contours2 in contours2:
                    (x, y), radius = cv2.minEnclosingCircle(i_contours2)
                    radius = int(radius)  # 半径，单位mm    int()向下取整，只取整数位
                    if opt.min_radius < radius < opt.max_radius:  # 半径范围
                        save_contours.append(i_contours2)  # 保存符合条件的轮廓

                # 产生符合条件的 mask  0-1
                result_contours_mask = np.zeros(([512, 512]))

                for i_save_contours in save_contours:
                    cv2.fillConvexPoly(result_contours_mask, i_save_contours, 1)  # 1 为填充值

                # mask -> bounding box
                lablel_mask = label(result_contours_mask)  # int64->uint8   0-3 -> 0-1
                props = regionprops(lablel_mask)

                result_dcm.append(i_2)
                for prop in props:
                    # 保存 bbox坐标(左上角坐标、右下角坐标)，与 标注信息 作对比
                    x_min, y_min, x_max, y_max = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
                    x_min = x_min - opt.expand_len / 2
                    y_min = y_min - opt.expand_len / 2
                    x_max = x_max + opt.expand_len / 2
                    y_max = y_max + opt.expand_len / 2

                    result_dcm.append(int(x_min))
                    result_dcm.append(int(y_min))
                    result_dcm.append(int(x_max))
                    result_dcm.append(int(y_max))

                    # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
                    cv2.rectangle(result, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 1)

                if not os.path.exists(opt.save_test_dir + str(i)):
                    os.makedirs(opt.save_test_dir + str(i))
                plt.imsave(opt.save_test_dir + str(i) + '/' + str(i_2) + '_' + 'result.jpg', result,
                           cmap=cm.gray)  # 仅有中间轮廓、且去掉肌肉的 且 过滤后存在bbox 的淋巴结范围的图像
                result_case.append(result_dcm)

            result_bbox.append(result_case)  # 一个病例结束

        return result_bbox

    def load_extract_img(self, dcm_dir):
        '''
        返回 提取轮廓的窗 0~255之间
        '''
        dcm = dicom.read_file(dcm_dir)
        img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        # 拿到 提取轮廓的窗，并规范化
        img_extract = (img_origin - (opt.WL_extract - opt.WW_extract / 2)) / opt.WW_extract * 255  # 规范化到0-255
        # 下界0，上界255
        return np.clip(img_extract, 0, 255)

    def load_node_img(self, dcm_dir):
        '''
        返回 提取淋巴结的窗 0~255之间
        '''
        dcm = dicom.read_file(dcm_dir)
        img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img_node = (img_origin - (opt.WL_node - opt.WW_node / 2)) / opt.WW_node * 255  # 规范化到0-255
        return np.clip(img_node, 0, 255)

    def get_contours(self, img_input):
        '''
        得到轮廓
        '''
        temp = np.uint8(img_input)  # uint8	无符号整数（0 到 255）
        ret, img = cv2.threshold(temp, 90, 255, cv2.THRESH_BINARY)  # 二值化
        _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
        return contours

    def get_center_contour(self, contours):
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
        max_contours_mask = np.zeros(([512, 512]))
        cv2.fillConvexPoly(max_contours_mask, contours[area_index[i]], 1)  # 1 为填充值
        return max_contours_mask
