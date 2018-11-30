import numpy as np
from config import opt
class getRecall():
    '''
    分组计算 两个list之间 召回率
    两组序列必须相同
    '''

    def __init__(self,predict_3D=0,truth_3D=0):
        self.predict_3D = predict_3D
        self.truth_3D = truth_3D

    def gruop_cal(self):
        '''
        分组计算
        '''
        result = []
        for i in range(len(self.predict_3D)):
            single_recall = self.single_cal(self.predict_3D[i], self.truth_3D[i])
            result.append(single_recall)
        # 得到召回率均值
        recall_mean = np.mean(np.array(result))
        return recall_mean

    def single_cal(self,single_predict_3D,single_truth_3D):
        '''
        单个序列下  计算两者  召回率
        '''

        # 标志位   若 真实淋巴结被框住，则对应位置置1
        flag_single_cal = np.zeros((len(single_truth_3D)))

        # 遍历真实淋巴结
        for i in range(len(single_truth_3D)):

            # 遍历 预测淋巴结，只要IOU匹配，则停止遍历，并将对应标志位置为1
            for single_predict_3D_coordinate in single_predict_3D:

                if self.iou_3d_cal(single_truth_3D[i],single_predict_3D_coordinate) >= opt.threshold_3d_iou:
                    # 认定框住
                    flag_single_cal[i] = 1
                    break


        # 匹配上的数目/总数目
        return sum(flag_single_cal)/flag_single_cal.size

    def iou_3d_cal(self,single_truth,single_predict):
        '''
        计算 两个 3D坐标 之间的 IOU
        '''


        A_x_min, A_y_min, A_z_min, A_x_max, A_y_max, A_z_max = single_truth[0],single_truth[1],single_truth[2],single_truth[3],single_truth[4],single_truth[5]
        B_x_min, B_y_min, B_z_min, B_x_max, B_y_max, B_z_max = single_predict[0],single_predict[1],single_predict[2],single_predict[3],single_predict[4],single_predict[5]

        # 体积
        A_volume = (A_x_max - A_x_min) * (A_y_max - A_y_min) * (A_z_max - A_z_min)
        B_volume = (B_x_max - B_x_min) * (B_y_max - B_y_min) * (B_z_max - B_z_min)

        # 遍历z，然后看每一层z的IOU
        overlap_area_list = []
        for i in range(min(A_z_min,B_z_min),max(A_z_max,B_z_max)+1):
            # 重叠面积
            if A_z_min <= i < A_z_max and B_z_min <= i < B_z_max:
                overlap_area = self.calcArea([A_x_min, A_y_min, A_x_max, A_y_max],[B_x_min, B_y_min, B_x_max, B_y_max])  # 重叠面积
                overlap_area_list.append(overlap_area)

        Inter_AB = sum(overlap_area_list)  # 交集体积
        Union_AB =  A_volume               # A_volume 真值体积

        return Inter_AB / (Union_AB + 0.0001)   # 防止 除数为0


    def calcArea(self,box_one, box_two):
        '''
        计算二维平面的IOU
        '''
        xmin_, ymin_, xmax_, ymax_ = box_one
        xmin, ymin, xmax, ymax = box_two


        # 判断
        if xmax_ < xmin_ or ymax_ < ymin_:
            print('box_one 坐标有问题===============')
        if xmax < xmin or ymax < ymin:
            print('box_two 坐标有问题===============')

        one_x, one_y, one_w, one_h = int((xmin_ + xmax_) / 2), int((ymin_ + ymax_) / 2), xmax_ - xmin_, ymax_ - ymin_
        two_x, two_y, two_w, two_h = int((xmin + xmax) / 2), int((ymin + ymax) / 2), xmax - xmin, ymax - ymin

        if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
            lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
            lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))
            rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
            rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))
            inter_w = abs(rd_x_inter - lu_x_inter)
            inter_h = abs(lu_y_inter - rd_y_inter)
            inter_square = inter_w * inter_h  # 相交面积
        else:
            inter_square = 0

        return inter_square