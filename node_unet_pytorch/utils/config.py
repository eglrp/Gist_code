import datetime
import os
class DefaultConfig():
    env = 'UNet_Node'  # visdom 环境的名字


    dataset_path='/home/bobo/data/NeckLymphNodes/dataset/'  # 数据集地址
    # 设置窗宽 窗位

    WL_extract, WW_extract = -360, 446  # 提取轮廓的窗宽 窗位
    WL_abdoment, WW_abdoment = 40, 350  # 腹窗的窗宽 窗位

    dcm_w=512
    dcm_h=512



    checkpoint_root='./checkpoints/'
    # load_model_path= None  # 加载预训练的模型的路径，为None代表不加载
    load_model_path= checkpoint_root + '99_unet.pth'  # 加载预训练的模型的路径，为None代表不加载

    use_gpu=True
    device_ids=[0,1] # 加载GPU的设备


    learning_rate=0.01
    train_percent = 0.9  # 训练集：验证集  =0.9 ：0.1
    max_epoch = 100
    milestones = [30,60,90]  # 学习率每到 数组内设定值即更新参数
    batch_size = 1
    num_workers = 0

    # =======验证==========
    use_dense_crf = True
    out_threshold = 0.5
    save_test_dir='./save_test/'


# 初始化该类的一个对象
opt = DefaultConfig()