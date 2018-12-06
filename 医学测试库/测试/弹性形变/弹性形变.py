import Augmentor
p = Augmentor.Pipeline('/home/bobo/data/test/test003/img')  # 实例化 Pipeline对象  地址为图片路径地址
p.ground_truth('/home/bobo/data/test/test003/mask')
# 向管道添加操作
p.random_distortion(probability=1,grid_width=3,grid_height=3,magnitude=3) # 需要调参
p.sample(10) # 生成10个样本



