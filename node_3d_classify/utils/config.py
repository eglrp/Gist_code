class DefaultConfig():
    env = 'Node_classify'  # visdom 环境的名字

    dataset_path = '/home/bobo/data/NeckLymphNodes/dataset/'  # 数据集地址



    WL_node, WW_node = 50, 60  # 淋巴结的窗宽 窗位

    use_gpu=True
    device_ids=[0,1] # 加载GPU的设备


    learning_rate=0.01
    train_percent = 0.9  # 训练集：验证集  =0.9 ：0.1
    max_epoch = 100
    milestones = [30,60,90]  # 学习率每到 数组内设定值即更新参数
    batch_size = 64  # 训练集的batch size
    num_workers = 4  # 加载数据时的线程数


    # 使用的模型，名字必须与models/__init__.py中的名字一致
    # 目前支持的网络
    model = 'Net_3D'



    # 存储模型的路径
    checkpoint_root =  '../checkpoint/'
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    # load_model_path = checkpoint_root+' .pth'




# 初始化该类的一个对象
opt = DefaultConfig()