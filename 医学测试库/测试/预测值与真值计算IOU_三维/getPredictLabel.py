import numpy as np
from config import opt
from getRecall import getRecall
class getPredictLabel:
    '''
    两组序列必须相同
    '''
    def __init__(self,predict_3D,truth_3D):
        self.predict_3D = predict_3D
        self.truth_3D = truth_3D
    def gruop_cal(self):
        '''
        分组计算
        '''
        result_label = []
        # 遍历 预测值
        for i in range(len(self.predict_3D)):
            single_label = self.single_cal(self.predict_3D[i], self.truth_3D[i])
            result_label.append(single_label)
        return result_label

    def single_cal(self, single_predict_3D, single_truth_3D):
        '''
        单个序列下  计算 预测3D块标签
        '''

        # 标志位   若 真实淋巴结被框住，则对应位置置1 (0为 负样本，1 为 正样本)
        flag_single_cal = np.zeros((len(single_predict_3D)))

        # 遍历预测淋巴结
        for i in range(len(single_predict_3D)):

            # 遍历 所有真实淋巴结，当 IOU最大值 超过阈值，则将对应标志位置为1
            temp = []
            for single_single_truth_3D_coordinate in single_truth_3D:

                temp.append(getRecall().iou_3d_cal(single_predict_3D[i],single_single_truth_3D_coordinate))
            if max(temp) >= opt.threshold_3d_iou:
                # 认定框住
                flag_single_cal[i] = 1
        return  flag_single_cal



