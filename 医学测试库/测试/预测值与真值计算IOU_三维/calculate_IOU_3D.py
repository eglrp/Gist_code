from getPredictCoordinate import getPredictCoordinate
from getTruthCoordinate import getTruthCoordinate
from getRecall import getRecall
from getPredictLabel import getPredictLabel
import numpy as np
def test():
    '''
    计算 4个病例的  预测值与 真值 在 三维上的IOU
    '''

    # 得到分组的list（预测值合成的3D块坐标  若bbox过多，则运行过慢 ）
    predict_3D = getPredictCoordinate().get_3d_coordinate()

    # 得到分组的list（ 真值3D块坐标）
    truth_3D = getTruthCoordinate().get_3d_coordinate()

    # # 分序列，得到召回率均值(两组序列必须相同)
    # value = getRecall(predict_3D,truth_3D).gruop_cal()
    # print(value)

    # 制作数据集
    # 保存 3D坐标
    np.save('all_predict_3d_coordinate.npy', predict_3D)
    np.save('all_truth_3d_coordinate.npy', truth_3D)

    # 保存 预测3D块 对应标签
    # 标准 : 当预测3D块 与真值 立体IOU>0.5，则认定为1
    predict_label = getPredictLabel(predict_3D, truth_3D).gruop_cal()
    np.save('predict_label.npy', predict_label)


if __name__ == '__main__':
    test()