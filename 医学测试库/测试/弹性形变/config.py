class DefaultConfig():
    dataset = '/home/bobo/data/NeckLymphNodes/newdataset2'

    abdomen_img_dir = '/home/bobo/data/test/test12/img'
    mask_dir = '/home/bobo/data/test/test12/mask'


    # 保存 弹性形变生成的结果
    train_tmp_path = '/home/bobo/data/test/test13/img'
    mask_tmp_path = '/home/bobo/data/test/test13/mask'

    # 保存最后结果
    train_result_path = '/home/bobo/data/test/test14/img'
    mask_result_path = '/home/bobo/data/test/test14/mask'

    WL_abdoment, WW_abdoment = 40, 350  # 腹窗的窗宽 窗位
    WL_extract, WW_extract = -360, 446  # 提取轮廓的窗宽 窗位


# 初始化该类的一个对象
opt = DefaultConfig()
