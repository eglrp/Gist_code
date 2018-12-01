from torch.utils import data
from utils.config import opt
import glob as gb
import numpy as np
class NodeDataSet(data.Dataset):
    '''
    主要目标： 根据训练，验证，测试划分数据
    '''

    def __init__(self,train=False,test=False,val=False):

        self.train = train
        self.test = test
        self.val = val

        # 加载 裁出来的3D块 及 对应标签
        block_3d,label = self.load_3d_block_label()

    def load_3d_block_label(self):
        '''
        加载 裁出来的3D块 及 对应标签
        '''
        # 加载dcm数据
        dcm_path = gb.glob(opt.dataset_path + 'dcm/*')
        dcm_path.sort()

        # 加载 预测3D块 及 对应标签
        [all_predict_3d_coordinate,all_predict_label] = np.load(opt.dataset_path + 'info/all_predict_info.npy')

        # 加载 真实3D块
        all_truth_3d_coordinate = np.load(opt.dataset_path + 'info/all_truth_3d_coordinate.npy')

        # 遍历每一个序列
        for i in range(len(dcm_path)):
            pass




def test():
    test_data = NodeDataSet(train=True)

if __name__ == '__main__':
    test()
