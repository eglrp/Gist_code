'''
弹性形变 增强数据
'''
import glob as gb
from 医学测试库.测试.弹性形变.config import opt
import numpy as np
import dicom
import Augmentor
import os
import random
import cv2
def get_dcm_dir():
    # 得到所有dcm路径
    dcm_list = gb.glob(opt.dataset + "/dcm/*/*.dcm")
    dcm_list.sort()
    return dcm_list


def get_mask():
    # 得到所有mask
    mask_list = gb.glob(opt.dataset + "/mask/*")
    mask_list.sort()
    return mask_list


def load_abdomen_img(dcm_dir):
    '''
    返回 腹窗 0~255之间
    '''
    dcm = dicom.read_file(dcm_dir)
    img_origin = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # 拿到 提取轮廓的窗，并规范化
    img_abdomen = (img_origin - (opt.WL_abdomen - opt.WW_abdomen / 2)) / opt.WW_abdomen * 255  # 规范化到0-255
    # 下界0，上界255
    return np.clip(img_abdomen, 0, 255)


def save_dcm_mask(dcm_dir, mask):
    '''
    dcm保存为腹窗
    格式均为jpg
    '''
    dcm_len = len(dcm_dir)
    # 保存 腹窗dcm
    for i in range(dcm_len):
        cv2.imwrite(opt.abdomen_img_dir + '/' + str(100000 + i) + '.jpg', load_abdomen_img(dcm_dir[i]))
        # pass
    # 保存 对应mask
    temp_mask = []
    for single_mask in mask:
        temp = np.load(single_mask)
        for j in range(temp.shape[0]):
            temp_mask.append(temp[j])
    for k in range(dcm_len):

        cv2.imwrite(opt.mask_dir + '/' + str(100000 + k) + '.jpg', temp_mask[k]*255)
        # pass


def format():
    train_img = gb.glob(opt.abdomen_img_dir + '/*.jpg')
    train_img.sort()
    masks = gb.glob(opt.mask_dir + '/*.jpg')
    masks.sort()

    if len(train_img) != len(masks):
        print("trains can't match masks")
        return 0

    for i in range(len(train_img)):

        # 新建一个文件夹，存储 一张图片
        train_img_tmp_path = opt.train_tmp_path + '/' + str(10000+i)
        if not os.path.exists(train_img_tmp_path):
            os.makedirs(train_img_tmp_path)
        # 加载图像并保存到 新建目录下
        image = cv2.imread(train_img[i], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(train_img_tmp_path + '/' + str(i) + '.jpg', image)

        # 新建一个文件夹，存储 一张对应mask
        mask_img_tmp_path = opt.mask_tmp_path + '/' + str(10000+i)
        if not os.path.exists(mask_img_tmp_path):
            os.makedirs(mask_img_tmp_path)
        # 加载图像并保存到 新建目录下
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(mask_img_tmp_path + '/' + str(i) + '.jpg', mask)

        # 创建了多少文件夹
        print("创建了 %s 个文件夹 !" % str(i))
    return len(train_img)


def doAugment(num):
    sum = 0
    for i in range(num):
        p = Augmentor.Pipeline(opt.train_tmp_path + '/' + str(10000+i))  # 拿到路径下的一张图片
        p.ground_truth(opt.mask_tmp_path + '/' + str(10000+i))  # 拿到对应路径的mask

        # 向管道添加操作
        # p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)  # 旋转
        # p.flip_left_right(probability=0.5)  # 按概率左右翻转
        # p.zoom_random(probability=0.6, percentage_area=0.99)  # 随即将一定比例面积的图形放大至全图
        # p.flip_top_bottom(probability=0.6)  # 按概率随即上下翻转
        # p.random_distortion(probability=0.8, grid_width=10, grid_height=10, magnitude=20)  # 小块变形
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=4)  # magnitude和grid_width,grid_height越大，扭曲程度越大

        count = random.randint(40, 60)  # 每张图对应的生成数量
        sum = sum + count
        p.sample(count)

    print("总共生成%s张" % sum)


def transform():
    '''
    读取生成的图像，img和mask分别保存到两个文件夹下
    '''

    array_img = gb.glob(opt.train_tmp_path + '/*/output')
    array_img.sort()

    for i in range(len(array_img)):
        # 遍历一个output文件夹

        # img和mask 后面哈希值一致
        array_origin = gb.glob(array_img[i] + '/*origin*.jpg')
        array_origin.sort()

        array_groundtruth = gb.glob(array_img[i] + '/*groundtruth*.jpg')
        array_groundtruth.sort()

        # 保存
        for j in range(len(array_origin)):
            cv2.imwrite(opt.train_result_path + '/'+str(i)+'_'+str(j) + '_origin.jpg', cv2.imread(array_origin[j], cv2.IMREAD_GRAYSCALE))
            cv2.imwrite(opt.mask_result_path + '/' + str(i) + '_' + str(j) + '_groundtruth.jpg',cv2.imread(array_groundtruth[j], cv2.IMREAD_GRAYSCALE))

def test():
    ###############################################
    # # 得到所有dcm路径
    # dcm_dir= get_dcm_dir()
    #
    # # 得到所有mask
    # mask= get_mask()
    #
    # # dcm及对应mask分别保存到两个文件夹
    # save_dcm_mask(dcm_dir,mask)

    ###############################################
    # # 开始弹性形变
    #
    # # 读取图像，创建文件夹仅保存一张图像及mask
    # img_len = format()
    # # 弹性形变
    # doAugment(img_len)
    #
    # # 总共生成11036张


    ##############################################3
    # 读取生成的图像，img和mask分别保存到两个文件夹下 (mask为0-255)
    transform()

if __name__ == '__main__':
    test()
