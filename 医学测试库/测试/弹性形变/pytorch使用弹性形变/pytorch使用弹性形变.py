import os
from torch.utils import data
import torch as t
import cv2
import glob as gb
import numpy as np
from utils.config import opt
import dicom
import random
from PIL import Image
from torch.utils.data import DataLoader
import Augmentor
import torchvision


class NodeDataSet(data.Dataset):
    '''
    主要目标： 根据训练，验证，测试划分数据
    '''

    def __init__(self, train=False):
        self.train = train
        # 训练、验证使用 弹性形变之后的数据（数据增强）

        train_img_dir = gb.glob('/home/bobo/data/test/test12/img' + "/*.jpg")
        train_img_dir.sort()

        train_mask_dir = gb.glob('/home/bobo/data/test/test12/mask' + "/*.jpg")
        train_mask_dir.sort()

        dcm_mask_list = list(zip(train_img_dir, train_mask_dir))
        self.dcm_mask_list = dcm_mask_list

        # 数据增强
        p = Augmentor.Pipeline()
        # p.flip_left_right(probability=1)  # ok
        p.crop_random(probability=1, percentage_area=0.9)  # ok  形状变小
        # p.rotate(probability=1.0, max_left_rotation=0,max_right_rotation=25)  #ok
        p.resize(probability=1.0, width=512, height=512)
        self.transforms = torchvision.transforms.Compose([
            p.torch_transform()
        ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''

        # 训练、验证使用 弹性形变之后的数据（数据增强）

        dcm_dir, mask_dir = self.dcm_mask_list[index]

        img_np = cv2.imread(dcm_dir, cv2.IMREAD_GRAYSCALE)  # 图像 [512,512]
        mask_np = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE) # 对应mask [512,512]
        null_np = np.zeros([512, 512], np.uint8) # 凑三通道 [512,512]

        # [512,512,1]
        img_np = t.unsqueeze(t.from_numpy(img_np), dim=-1)
        mask_np = t.unsqueeze(t.from_numpy(mask_np), dim=-1)
        null_np = t.unsqueeze(t.from_numpy(null_np), dim=-1)

        # [512,512,3]
        result_np = t.cat((img_np, mask_np, null_np), -1).numpy()

        # cv2.imwrite('1bobo_0.jpg', result_np[:, :, 0])
        # cv2.imwrite('1bobo_1.jpg', result_np[:, :, 1])
        result_np = Image.fromarray(result_np)

        # 数据增强
        result_np= self.transforms(result_np)
        result_np = np.array(result_np)
        # cv2.imwrite('2bobo_0.jpg', result_np[:, :, 0])
        # cv2.imwrite('2bobo_1.jpg', result_np[:, :, 1])
        print()
        return 1,2

    def __len__(self):
        return len(self.dcm_mask_list)


# ===============================================================================
def test():
    train_data = NodeDataSet(train=True)
    train_dataloader = DataLoader(train_data, 2, shuffle=True, num_workers=1)
    for ii, (block_3d, truth_label) in enumerate(train_dataloader):
        print()


if __name__ == '__main__':
    test()
