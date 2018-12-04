from torch.utils import data
from utils.config import opt
import glob as gb
import numpy as np
import random
import dicom
import torch as t
from torchvision import transforms as T


class NodeDataSet(data.Dataset):
    '''
    主要目标： 根据训练，验证，测试划分数据
    '''

    def __init__(self, train=False, test=False, val=False):

        self.train = train
        self.test = test
        self.val = val

        # 加载数据集
        all_block_3d = self.load_3d_block_label()
        # 打乱数据集
        random.shuffle(all_block_3d)

        block_num = len(all_block_3d)
        # 测试集
        if self.train:
            self.block = all_block_3d[:int(opt.train_percent * block_num)]
        if self.val:
            self.block = all_block_3d[int(opt.train_percent * block_num):]
        if self.test:
            pass

        # # 规范化
        # self.transforms = T.Compose([
        #     T.ToTensor(), # 变维度，不可用
        # ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        sequence_index, coordinate_3d, label = self.block[index]
        # 通过 序列号、3d坐标 取出对应的3D块
        block_3d = self.get_3d_block(sequence_index, coordinate_3d)
        # 数据增强=============================================

        block_3d = (block_3d - np.min(block_3d)) / (np.max(block_3d) - np.min(block_3d))  # min-max标准化   归一化

        block_3d = t.unsqueeze(t.from_numpy(block_3d), dim=0)  # [channels,depth,h,w]
        return block_3d, label

    def __len__(self):
        '''
        返回数据集的总数
        '''
        return len(self.block)

    def load_3d_block_label(self):
        '''
        加载 裁出来的3D块 及 对应标签
        '''

        # 加载 预测3D块 及 对应标签
        [all_predict_3d_coordinate, all_predict_label] = np.load(opt.dataset_path + 'info/all_predict_info.npy')

        # 加载 真实3D块
        all_truth_3d_coordinate = np.load(opt.dataset_path + 'info/all_truth_3d_coordinate.npy')

        # list包含 [
        #     [[序列号],[3D坐标]，[标签]] 一个3D块
        #   ]
        all_block_3d = []

        # 将预测值写入list
        for i_predict in range(len(all_predict_3d_coordinate)):
            for j_predict in range(len(all_predict_3d_coordinate[i_predict])):
                temp_predict = []
                temp_predict.append(i_predict)  # 序列号
                temp_predict.append(all_predict_3d_coordinate[i_predict][j_predict])  # 3D坐标
                temp_predict.append(int(all_predict_label[i_predict][j_predict]))  # 标签
                all_block_3d.append(temp_predict)
        # 将 真实标注 写入list,所有标签都为1

        for i_truth in range(len(all_truth_3d_coordinate)):
            for single_truth in all_truth_3d_coordinate[i_truth]:
                temp_truth = []
                temp_truth.append(i_truth)  # 序列号
                temp_truth.append(single_truth)  # 3D坐标
                temp_truth.append(1)  # 标签
                all_block_3d.append(temp_truth)
        return all_block_3d

    def get_3d_block(self, sequence_index, coordinate_3d):
        '''
        通过 序列号、3d坐标 取出对应的3D块
        '''
        dcm_path = gb.glob(opt.dataset_path + 'dcm/*')
        dcm_path.sort()  # 排序

        single_dcm_path = gb.glob(dcm_path[sequence_index] + '/*.dcm')
        single_dcm_path.sort()

        dcm_path_range = single_dcm_path[coordinate_3d[2]:coordinate_3d[5] + 1]
        block_3d = []
        for i_path in dcm_path_range:
            # 加载为 淋巴结范围的图像 0~255
            img_node = self.load_node_range(i_path)
            block_3d.append(img_node[coordinate_3d[1]:coordinate_3d[4] + 1, coordinate_3d[0]:coordinate_3d[3] + 1])
        return np.array(block_3d)

    def load_node_range(self, i_path):
        '''
        加载为 淋巴结范围的图像
        '''
        # 读取dcm文件
        dcm = dicom.read_file(i_path)
        img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

        # 拿到 提取轮廓的窗，并规范化
        img_node = (img_origin - (opt.WL_node - opt.WW_node / 2)) / opt.WW_node * 255  # 规范化到0-255
        return np.clip(img_node, 0, 255)

# def test():
#     NodeDataSet(train=True)
#
# if __name__ == '__main__':
#     test()
