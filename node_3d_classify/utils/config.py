class DefaultConfig():
    env = 'Node_classify_'  # visdom 环境的名字

    dataset_path = '/home/bobo/data/NeckLymphNodes/newdataset/'  # 数据集地址

    WL_node, WW_node = 50, 60  # 淋巴结的窗宽 窗位

    use_gpu = True
    device_ids = [0, 1]  # 加载GPU的设备

    # 目前支持的网络
    model = 'Net_3D'
    # 块池化方式（默认平均池化）
    block_max_pooling = True

    # 存储模型的路径
    checkpoint_root = '../checkpoint/'
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    # load_model_path = checkpoint_root+' .pth'

    # 训练参数
    learning_rate = 0.01
    weight_decay = 0e-5  # 损失函数
    train_percent = 0.8  # 训练集：验证集  =0.8 ：0.2
    max_epoch = 200
    milestones = [40, 80, 120, 160]  # 学习率每到 数组内设定值即更新参数
    batch_size = 4  # 训练集的batch size
    num_workers = 4  # 加载数据时的线程数


# 初始化该类的一个对象
opt = DefaultConfig()
