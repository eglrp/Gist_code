import models as models
from utils.config import opt
from utils.visualize import Visualizer
import torch as t
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from dataloader.NodeDataSet import NodeDataSet
from torch.utils.data import DataLoader
import os
def train():
    vis = Visualizer(opt.env + opt.model)
    net = getattr(models, opt.model)()
    print('当前使用的模型为' + opt.model)
    # 分类损失函数使用交叉熵
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    start_epoch = 0
    if opt.load_model_path:
        checkpoint = t.load(opt.load_model_path)

        # 加载多GPU模型参数到 CPU上
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)                 # 加载模型
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器
        start_epoch = checkpoint['epoch']                   # 加载训练批次

    # 学习率每当到达milestones值则更新参数
    if start_epoch == 0:
        scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1,
                                                     last_epoch=-1)
        print('从头训练 ，学习率为{}'.format(optimizer.param_groups[0]['lr']))
    else:
        scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1,
                                                     last_epoch=start_epoch)
        print('加载预训练模型{}并从{}轮开始训练,学习率为{}'.format(opt.load_model_path, start_epoch, optimizer.param_groups[0]['lr']))

    # 网络转移到GPU上
    if opt.use_gpu:
        net = t.nn.DataParallel(net, device_ids=opt.device_ids)  # 模型转为GPU并行
        net.cuda()
        cudnn.benchmark = True

    train_data = NodeDataSet(train=True)
    val_data = NodeDataSet(val=True)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    for epoch in range(opt.max_epoch - start_epoch):
        print('开始 epoch {}/{}.'.format(start_epoch + epoch + 1, opt.max_epoch))
        epoch_loss = 0

        # 每轮判断是否更新学习率
        scheduler.step()

        # 迭代数据集加载器
        for ii, (block_3d, truth_label) in enumerate(train_dataloader):
            if opt.use_gpu:
                block_3d = block_3d.cuda()
                truth_label = truth_label.cuda()
            predict_label = net(block_3d)

            loss = criterion(predict_label, truth_label)
            epoch_loss += loss.item()

            if ii % 8 == 0:
                vis.plot('训练集loss', loss.item())

            optimizer.zero_grad()  # 优化器梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        # 当前时刻的一些信息
        vis.log("epoch:{epoch},lr:{lr},loss:{loss}".format(
            epoch=epoch, loss=loss.item(), lr=optimizer.param_groups[0]['lr']))
        vis.plot('每轮epoch的loss均值', epoch_loss / ii)
        # 保存模型、优化器、当前轮次等
        state = {'net':net.state_dict(),'optimizer':optimizer.state_dict(), 'epoch':epoch}
        if not os.path.exists(opt.checkpoint_root):
            os.makedirs(opt.checkpoint_root)
        t.save(state,opt.checkpoint_root + '{}_node.pth'.format(epoch))

        # ============验证===================
        val_loss = 0
        with t.no_grad():
            for jj, (val_block_3d, val_label) in enumerate(val_dataloader):
                if opt.use_gpu:
                    val_block_3d = val_block_3d.cuda()
                    val_label = val_label.cuda()
                val_predict_label = net(val_block_3d)
                loss = criterion(val_predict_label, val_label)
                val_loss += loss.item()
            vis.plot('验证集loss均值', val_loss / jj)



if __name__ == '__main__':
    #开始训练
    train()