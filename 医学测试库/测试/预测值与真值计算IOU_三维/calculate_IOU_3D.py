'''
计算 4个病例的  预测值与 真值 在 三维上的IOU
'''
from getPredictCoordinate import getPredictCoordinate
from getTruthCoordinate import getTruthCoordinate
from getRecall import getRecall
def test():
    # 使用的文本是假数据

    # 得到分组的list（预测值合成的3D块坐标  若bbox过多，则运行过慢 ）
    predict_3D = getPredictCoordinate().get_3d_coordinate()

    # 得到分组的list（ 真值3D块坐标）
    truth_3D = getTruthCoordinate().get_3d_coordinate()


    # 分序列，得到召回率均值(两组序列必须相同)
    value = getRecall(predict_3D,truth_3D).gruop_cal()
    print('===========================')
    print(value)

if __name__ == '__main__':
    test()