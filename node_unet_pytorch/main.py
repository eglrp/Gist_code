from models import UNet
import torch as t
from utils.config import opt
import torch.backends.cudnn as cudnn
from dataloader.NodeDataSet import NodeDataSet
from torch.utils.data import DataLoader
from utils.dice_loss import dice_loss
from matplotlib import cm
import matplotlib.pyplot as plt
from utils.visualize import Visualizer
from collections import OrderedDict
import numpy as np
def train():



    # n_channels：数医学影像为一通道灰度图    n_classes：二分类
    net = UNet(n_channels=1, n_classes=1)
    optimizer = t.optim.SGD(net.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=0.0005)
    criterion = t.nn.BCELoss()  # 二进制交叉熵(适合mask占据图像面积较大的场景)


    start_epoch = 0
    if opt.load_model_path:
        checkpoint = t.load(opt.load_model_path)

        # 加载多GPU模型参数到 单模型上
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)  # 加载模型
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器
        start_epoch = checkpoint['epoch']  # 加载训练批次


    # 学习率每当到达milestones值则更新参数
    if start_epoch == 0:
        scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1, last_epoch=-1) # 默认为-1
        print('从头训练 ，学习率为{}'.format(optimizer.param_groups[0]['lr']))
    else:
        scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1, last_epoch=start_epoch)
        print('加载预训练模型{}并从{}轮开始训练,学习率为{}'.format(opt.load_model_path, start_epoch, optimizer.param_groups[0]['lr']))

    # 网络转移到GPU上
    if opt.use_gpu:
        net = t.nn.DataParallel(net,device_ids=opt.device_ids)  # 模型转为GPU并行
        net.cuda()
        cudnn.benchmark = True


    # 定义可视化对象
    vis = Visualizer(opt.env)

    train_data = NodeDataSet(train=True)
    val_data = NodeDataSet(val=True)

    # 数据集加载器
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    for epoch in range(opt.max_epoch - start_epoch):
        print('开始 epoch {}/{}.'.format(start_epoch + epoch + 1, opt.max_epoch))
        epoch_loss = 0

        # 每轮判断是否更新学习率
        scheduler.step()

        # 迭代数据集加载器
        for ii, (img,mask) in enumerate(train_dataloader):  #pytorch0.4写法，不再将tensor封装为Variable
            # 将数据转到GPU
            if opt.use_gpu:
                img = img.cuda()
                true_masks = mask.cuda()
            masks_pred = net(img)

            # 经过sigmoid
            masks_probs = t.sigmoid(masks_pred)

            # 损失 = 二进制交叉熵损失 + dice损失
            loss = criterion(masks_probs.view(-1), true_masks.view(-1))
            loss += dice_loss(masks_probs, true_masks)
            epoch_loss += loss.item()

            if ii % 8 == 0:
                vis.plot('训练集loss', loss.item())

            # 优化器梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()




        # 当前时刻的一些信息
        vis.log("epoch:{epoch},lr:{lr},loss:{loss}".format(
            epoch=epoch, loss=loss.item(), lr=optimizer.param_groups[0]['lr']))

        vis.plot('每轮epoch的loss均值', epoch_loss / ii)
        # 保存模型、优化器、当前轮次等
        state = {'net':net.state_dict(),'optimizer':optimizer.state_dict(), 'epoch':epoch}
        t.save(state,opt.checkpoint_root + '{}_unet.pth'.format(epoch))


        # ============验证===================

        # 评价函数：Dice系数    Dice距离用于度量两个集合的相似性
        tot = 0
        for jj, (img_val,mask_val) in enumerate(val_dataloader):
            img_val =img_val
            true_mask_val = mask_val
            if opt.use_gpu:
                img_val = img_val.cuda()
                true_mask_val = true_mask_val.cuda()

            mask_pred = net(img_val)
            mask_pred = (t.sigmoid(mask_pred) > 0.5).float()   # 阈值为0.5
            # 评价函数：Dice系数   Dice距离用于度量两个集合的相似性
            tot += dice_loss(mask_pred, true_mask_val).item()
        val_dice= tot / jj
        vis.plot('验证集 Dice损失', val_dice)

def predict():
    net = UNet(n_channels=1, n_classes=1)

    # 将多GPU模型加载为CPU模型
    if opt.load_model_path:
        checkpoint = t.load(opt.load_model_path)
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)  # 加载模型
        print('加载预训练模型{}'.format(opt.load_model_path))
    if opt.use_gpu:
        net.cuda()

    test_data = NodeDataSet(test=True)

    test_dataloader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    for ii, full_img in enumerate(test_dataloader):
        img_test=full_img[0][0].unsqueeze(0)  # 第一个[0]  取 原图像的一个batch，第二个[0]指batch为1
        if opt.use_gpu:
            img_test =img_test.cuda()

        with t.no_grad(): #pytorch0.4版本写法
            output = net(img_test)
            probs = t.sigmoid(output).squeeze(0)

        full_mask = probs.squeeze().cpu().numpy()
        # ===========================================下面方法可能未考虑 一通道图像
        # if opt.use_dense_crf:
        #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)
        mask = full_mask > opt.out_threshold    #预测mask值都太小,最大0.01

        # 可视化1
        # plt.imsave(opt.save_test_dir+str(10000+ii)+'full_img.jpg', full_img[0][0].squeeze(0),cmap = cm.gray)  #保存原图
        # plt.imsave(opt.save_test_dir+str(10000+ii)+'mask.jpg', mask,cmap = cm.gray) #保存mask
        # plt.imsave(opt.save_test_dir+str(10000+ii)+'full_mask.jpg', full_img[0][0].squeeze(0).squeeze(0).numpy() * mask,cmap = cm.gray)  #保存mask之后的原图

        # 可视化2
        # # 多子图显示原图和mask
        # plt.subplot(1,3,1)
        # plt.title('origin')
        # plt.imshow(full_img[0][0].squeeze(0),cmap='Greys_r')
        #
        # plt.subplot(1, 3, 2)
        # plt.title('mask')
        # plt.imshow(mask,cmap='Greys_r')
        #
        # plt.subplot(1, 3, 3)
        # plt.title('origin_after_mask')
        # plt.imshow( full_img[0][0].squeeze(0).squeeze(0).numpy() * mask,cmap='Greys_r')
        #
        # plt.show()

        # 保存mask为npy
        np.save('/home/bobo/data/test/mask_result/' + str(10000+ii) + '_mask.npy',mask)

    print('测试完毕')

if __name__ == '__main__':
    # pytorch0.4

    # 训练
    # train()

    # 预测
    predict()