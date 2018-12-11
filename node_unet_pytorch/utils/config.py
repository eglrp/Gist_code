import datetime
import os


class DefaultConfig():
    env = 'UNet_Node'  # visdom 环境的名字

    dataset_path = '/home/bobo/data/NeckLymphNodes/newdataset2/'  # 数据集地址

    # 弹性形变的数据
    abdomen_img_dir = '/home/bobo/data/test/test14/img'
    mask_dir = '/home/bobo/data/test/test14/mask'

    xml_dir = '/home/bobo/data/test/bbox_xml/'  # 真值淋巴结xml

    # 设置窗宽 窗位
    WL_extract, WW_extract = -360, 446  # 提取轮廓的窗宽 窗位
    WL_abdoment, WW_abdoment = 40, 350  # 腹窗的窗宽 窗位
    WL_node, WW_node = 50, 60  # 淋巴结的窗宽 窗位

    expand_len = 20  # 以轮廓为中心， 每层的bbox 至少长宽为10 （将长宽不足10的 补到10x10）
    # 最小外接圆直径以内的轮廓 转为 bbox  半径范围
    min_radius = 0
    max_radius = 20

    checkpoint_root = './checkpoints/'
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    # load_model_path= checkpoint_root + '99_unet.pth'  # 加载预训练的模型的路径，为None代表不加载

    use_gpu = True
    device_ids = [1]  # 加载GPU的设备

    learning_rate = 0.01
    train_percent = 0.8  # 训练集：验证集  =0.8 ：0.2
    max_epoch = 200
    milestones = [40, 80, 120, 160]  # 学习率每到 数组内设定值即更新参数
    batch_size = 1  # 训练 and 验证
    test_batch_size = 1  # 预测
    num_workers = 4

    # =========网络结构===========
    use_BN = False
    use_dice_loss = False  # 损失 是否加入dice

    # =======验证==========
    use_dense_crf = True
    out_threshold = 0.5
    save_test_dir = '/home/bobo/data/test/test7/'

    # ==========测试========
    iou_threshold = 0.8  # 二维IOU阈值


# 初始化该类的一个对象
opt = DefaultConfig()
