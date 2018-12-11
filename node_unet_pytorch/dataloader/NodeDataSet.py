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


class NodeDataSet(data.Dataset):
    '''
    主要目标： 根据训练，验证，测试划分数据
    '''

    def __init__(self, train=False, test=False, val=False):

        self.train = train
        self.test = test
        self.val = val

        # 训练、验证使用 弹性形变之后的数据（数据增强）
        if not self.test:
            train_img_dir = gb.glob(opt.abdomen_img_dir + "/*.jpg")
            train_img_dir.sort()

            train_mask_dir = gb.glob(opt.mask_dir + "/*.jpg")
            train_mask_dir.sort()

            dcm_mask_list = list(zip(train_img_dir, train_mask_dir))

            # 打乱数据集
            random.shuffle(dcm_mask_list)

            # 划分数据集
            if self.train:  # 训练集
                self.dcm_mask_list = dcm_mask_list[:int(opt.train_percent * len(dcm_mask_list))]

            if self.val:  # 验证集
                self.dcm_mask_list = dcm_mask_list[int(opt.train_percent * len(dcm_mask_list)):]

        else:
            # 测试使用 原始数据（不数据增强）

            # 读取所有dcm
            dcm_path_dicom = gb.glob(opt.dataset_path + "dcm/*/*.dcm")
            dcm_path_dicom.sort()  # 排序

            # 读取所有mask
            mask_path_dicom = gb.glob(opt.dataset_path + "mask/*.npy")
            mask_path_dicom.sort()  # 排序

            # 将dcm和mask一一对应
            mask_np = np.zeros((0, 512, 512))  # 以便合并，该tensor无用  512为CT的长宽
            for i in range(len(mask_path_dicom)):
                mask_np = np.concatenate((mask_np, np.load(mask_path_dicom[i])), axis=0)
            self.dcm_mask_list = list(zip(dcm_path_dicom, mask_np))

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''

        # 训练、验证使用 弹性形变之后的数据（数据增强）
        if not self.test:
            dcm_dir, mask_dir = self.dcm_mask_list[index]

            img_np = cv2.imread(dcm_dir, cv2.IMREAD_GRAYSCALE)
            mask_np = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)

            img_np = np.array(img_np, dtype=np.float32) / 255  # [512,512]  0~1
            mask_np = np.array(mask_np, dtype=np.float32) / 255  # [512,512]  0-1 mask

            img_np, mask_np = self.DataAugmentation(img_np, mask_np)  # 数据增强
        else:
            # 测试使用 原始数据（不数据增强）
            dcm_dir, mask_np = self.dcm_mask_list[index]
            # 提取 仅保留中间连通区域的腹窗图像
            dcm_np = self.Pretreatment(dcm_dir)

            img_np = np.array(dcm_np, dtype=np.float32) / 255  # [512,512]  0~1
            mask_np = np.array(mask_np, dtype=np.float32)  # [512,512]  0-1 mask

        # 转化为torch  tensor  [512,512]转为[1,512,512]
        img = t.unsqueeze(t.from_numpy(img_np), dim=0)
        mask = t.from_numpy(mask_np)
        return img, mask

    def __len__(self):
        return len(self.dcm_mask_list)

    def Pretreatment(self, dcm_dir):
        '''
        仅保留中间连通区域的腹窗图像
        '''
        # 读取dcm
        dcm = dicom.read_file(dcm_dir)
        img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

        # 拿到 提取轮廓的窗，并规范化
        img_extract = (img_origin - (opt.WL_extract - opt.WW_extract / 2)) / opt.WW_extract * 255  # 规范化到0-255
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

        # 拿到 腹窗
        img_abdoment = (img_origin - (opt.WL_abdoment - opt.WW_abdoment / 2)) / opt.WW_abdoment * 255  # 规范化到0-255
        img_abdoment[img_abdoment < 0] = 0
        img_abdoment[img_abdoment > 255] = 255

        return img_abdoment * max_contours_mask

    def DataAugmentation(self, img_np, mask_np):
        '''
        0.5概率 镜像翻转
        0.5概率  旋转正负30度
        :param img_np: [512,512]
        :param mask_np: [512,512]
        '''

        # 合并img和mask，统一增强
        mix_np = np.concatenate((np.expand_dims(img_np, axis=0), np.expand_dims(mask_np, axis=0)),
                                axis=0)  # [2,512,512]
        mix_np = self.random_flip(mix_np)  # 随机镜像翻转
        mix_np = self.random_rotate(mix_np)  # 随机旋转正负30度以内

        return mix_np[0], mix_np[1]

    def random_flip(self, im):
        '''
        随机镜像翻转
        '''

        # #原版可视化
        # plt.subplot(1, 4, 1)
        # plt.imshow(im[0])
        #
        # plt.subplot(1, 4, 2)
        # plt.imshow(im[1])

        if random.random() < 0.5:
            # 双通道比较特殊，只能拆开 再进行翻转
            im_lr_0 = np.fliplr(im[0]).copy()  # 左右翻转
            im_lr_1 = np.fliplr(im[1]).copy()  # 左右翻转

            # #旋转后 可视化
            # plt.subplot(1, 4, 3)
            # plt.imshow(im_lr_0)
            #
            # plt.subplot(1, 4, 4)
            # plt.imshow(im_lr_1)
            #
            # plt.show()

            im_lr = np.concatenate((np.expand_dims(im_lr_0, axis=0), np.expand_dims(im_lr_1, axis=0)), axis=0)
            return im_lr
        return im

    def random_rotate(self, im):
        '''
        随机旋转正负30度
        '''
        # #原版可视化
        # plt.subplot(1, 4, 1)
        # plt.imshow(im[0])
        #
        # plt.subplot(1, 4, 2)
        # plt.imshow(im[1])

        if random.random() < 0.5:
            # 双通道比较特殊，只能拆开 再进行翻转
            angle = random.randint(-30, 30)  # 随机产生-30 ~ 30的随机数用于旋转

            im_0 = Image.fromarray(im[0])  # numpy类型转化为Image类型
            im_rotate_0 = im_0.rotate(angle)

            im_1 = Image.fromarray(im[1])  # numpy类型转化为Image类型
            im_rotate_1 = im_1.rotate(angle)

            im_rotate0 = np.array(im_rotate_0)
            im_rotate1 = np.array(im_rotate_1)

            # #旋转后 可视化
            # plt.subplot(1, 4, 3)
            # plt.imshow(im_rotate0)
            #
            # plt.subplot(1, 4, 4)
            # plt.imshow(im_rotate1)
            #
            # plt.show()

            im_rotate = np.concatenate((np.expand_dims(im_rotate0, axis=0), np.expand_dims(im_rotate1, axis=0)), axis=0)
            return im_rotate
        return im
