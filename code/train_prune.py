import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model_mobileNetV1_prune import tiny_detector, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import os
from torchsummary import summary
from torch import nn
import numpy as np

###################################################
# 注意，train.py要在tiny_detector_demo这个文件夹下运行 #
###################################################

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # 无关紧要，控制pc运行哪块gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 控制设备，读取gpu还是cpu
cudnn.benchmark = True

# Data parameters
data_folder = '../../../dataset/VOCdevkit/'  # data files root path
keep_difficult = True  # use objects considered difficult to detect? ， 目标中是否有复杂的
n_classes = len(label_map)  # number of different types of objects，定义了模型的类别数， label_map是在utils里定义的

# Learning parameters
total_epochs = 60  # number of epochs to train
prune_epochs = 40
batch_size = 4  # batch size
workers = 2  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [150, 190]  # decay learning rate after these many epochs
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay

# 剪枝相关参数
layers = 25  # BN层的层数
prune_percent = 0.5  # 剪枝率
base_number = 1


def main():
    """
    Training.
    """
    min_losses = 10
    # Initialize model and optimizer
    # 初始化模型，参数是类别数
    model = tiny_detector(n_classes=n_classes)  # 初始化模型
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)  # 初始化损失函数
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=lr, momentum=momentum, weight_decay=weight_decay)  # 初始化优化器， 用的是SGD，随机梯度下降

    # Move to default device
    model = model.to(device)  # 把模型放到gpu进行计算
    summary(model, (3, 224, 224))
    criterion = criterion.to(device)  # 把损失函数放到gpu进行计算

    # Custom dataloaders， 设置 dataset 和 dataloader
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    ####################################  初次训练  ######################################
    # Epochs
    for epoch in range(total_epochs):
        # Decay learning rate at particular epochs， 要在150 和 190 个周期后降低学习率
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        losses = train(train_loader=train_loader,
                       model=model,
                       criterion=criterion,
                       optimizer=optimizer,
                       epoch=epoch)

        # Save checkpoint，每个epoch 保存一次
        if losses < min_losses:
            min_losses = losses
            save_checkpoint_withtime(epoch, model, optimizer, losses)
    print(f"模型已训练完成")

    ####################################  预剪枝  ######################################
    print(f"开始模型预剪枝")
    cfg_0, cfg, cfg_mask = pre_prune(epoch, model, optimizer, losses)
    print(f"模型预剪枝完成")
    # summary(model, (3, 224, 224))

    ####################################  正式剪枝  ######################################
    print(f"正式剪枝开始")
    newmodel = prune(epoch, model, optimizer, losses, cfg_0, cfg, cfg_mask)
    summary(newmodel, (3, 224, 224))
    print(f"正式剪枝完成")

    print(f"开始重训练")
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    criterion1 = MultiBoxLoss(priors_cxcy=newmodel.priors_cxcy)  # 初始化损失函数
    optimizer1 = torch.optim.SGD(params=newmodel.parameters(),
                                 lr=lr, momentum=momentum, weight_decay=weight_decay)  # 初始化优化器， 用的是SGD，随机梯度下降
    criterion1 = criterion1.to(device)  # 把损失函数放到gpu进行计算
    min_losses = 10
    # Epochs
    ####################################  再次训练  ######################################
    for epoch in range(prune_epochs):
        # Decay learning rate at particular epochs， 要在150 和 190 个周期后降低学习率
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        losses = train(train_loader=train_loader,
                       model=newmodel,
                       criterion=criterion1,
                       optimizer=optimizer1,
                       epoch=epoch)

        # Save checkpoint，每个epoch 保存一次
        if losses < min_losses:
            min_losses = losses
            save_checkpoint_withprune(epoch, newmodel, optimizer1, losses)
    print(f"重训练完成")


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 224, 224)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 441, 4), (N, 441, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return losses.avg


# 进行预剪枝和早期处理工作
def pre_prune(epoch, model, optimizer, losses):
    print(f"3.test model\n")

    # 统计BN层的通道数
    total = 0  # BN层通道数
    i = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                total += m.weight.data.shape[0]
                i += 1
                print(f"the {i} layers size is {m.weight.data.shape[0]}")
    print(f"4.Batch Normalizational layers = {i}")
    print(f"5.Total BN channels is {total}")

    # 确定剪枝的全局阈值
    bn_shadow = torch.zeros(total)
    print(f"6.empty total bn shadow is {bn_shadow}")
    index = 0
    i = 0
    # 给出BN层的通道映射
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                size = m.weight.data.shape[0]
                bn_shadow[index:(index + size)] = m.weight.data.abs().clone()
                index += size
    print(f"7.new BN shadow is {bn_shadow}")
    print(f"8.new BN shadow size is {bn_shadow.size()}")

    # 按照权值大小得到阈值
    y, j = torch.sort(bn_shadow)
    thre_index = int(total * prune_percent)
    if thre_index == total:
        thre_index = total - 1
    thre_number = y[thre_index]
    print(f"9.index of thread is {thre_index}, & thread number is {thre_number}")

    # 预剪枝
    pruned = 0
    cfg_0 = []
    cfg = []
    cfg_mask = []
    i = 0

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                weight_copy = m.weight.data.clone()
                mask = weight_copy.abs().gt(thre_number).float()  # 大于阈值的加入mask
                remain_channels = torch.sum(mask)

                # 如果全部剪掉的话就提示应该调小剪枝程度了， remain_channels = 这一层保留的通道数
                if remain_channels == 0:  # 全都小于阈值
                    print('\r\n!please turn down the prune_ratio!\r\n')
                    remain_channels = 1
                    mask[int(torch.argmax(weight_copy))] = 1  # 这一层里最大的哪个保留下来，置为1

                # 规整剪枝
                v = 0
                n = 1
                if remain_channels % base_number != 0:
                    if remain_channels > base_number:
                        while v < remain_channels:
                            n += 1
                            v = base_number * n
                        if remain_channels - (v - base_number) < v - remain_channels:
                            remain_channels = v - base_number
                        else:
                            remain_channels = v
                        if remain_channels > m.weight.data.size()[0]:
                            remain_channels = m.weight.data.size()[0]
                        remain_channels = torch.tensor(remain_channels)

                        y, j = torch.sort(weight_copy.abs())
                        thre_1 = y[-remain_channels]
                        mask = weight_copy.abs().ge(thre_1).float()

                # 剪枝掉的通道个数
                pruned = pruned + mask.shape[0] - torch.sum(mask)  # mask.shape[0]:这一层的总通道数， torch.sum(mask): 这一层剩下的通道数
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg_0.append(mask.shape[0])
                cfg.append(int(remain_channels))
                cfg_mask.append(mask.clone())
                print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                      format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))

    # 预剪枝完成
    pruned_ratio = float(pruned / total)
    print('\r\n10.!预剪枝完成!')
    print('total_pruned_ratio: ', pruned_ratio)
    print(f"11.cfg = {cfg}")
    # print(f"12.cfg_mask = {cfg_mask}")
    # save_checkpoint_withprune(epoch, model, optimizer, losses)
    return cfg_0, cfg, cfg_mask


def prune(epoch, model, optimizer, losses, cfg_0, cfg, cfg_mask):
    # cfg:  BN层保留的通道数
    # cfg_mask: BN层保留的mask
    newModel = tiny_detector(n_classes=n_classes, cfg=cfg)
    newModel.cuda()
    layer_id_in_cfg = 0
    startMask = torch.ones(3)
    endMask = cfg_mask[layer_id_in_cfg]
    i = 0

    for [m0, m1] in zip(model.modules(), newModel.modules()):  # 分别从新模型和老模型中提取
        if isinstance(m0, nn.BatchNorm2d):
            if i < layers - 1:
                # np.argwhere 返回返回非0的数组元组的索引，其中括号内是要索引数组的条件
                # idx1 = 保留的元素的位置
                i += 1
                idx1 = np.squeeze(
                    np.argwhere(np.asarray(endMask.cpu().numpy())))  # squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
                print(f"13.idx1.size = {idx1.size}")
                print(f"14.idx1 = {idx1}")
                print(f"15.i = {i}")
                print(f"16.BN层")
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                layer_id_in_cfg += 1
                startMask = endMask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    endMask = cfg_mask[layer_id_in_cfg]
            # else:
            #     m1.weight.data = m0.weight.data[idx1].clone()
            #     m1.bias.data = m0.bias.data[idx1].clone()
            #     m1.running_mean = m0.running_mean[idx1].clone()
            #     m1.running_var = m0.running_var[idx1].clone()
        elif isinstance(m0, nn.Conv2d):  # m0 老模型
            if i < layers - 1:
                print(f"17.Conv层")
                idx0 = np.squeeze(np.argwhere(np.asarray(startMask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(endMask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx0, (1,))
                w = m0.weight.data[:, idx0, :, :].clone()
                m1.weight.data = w[idx1, :, :, :].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
            # else:
            #     idx0 = np.squeeze(np.argwhere(np.asarray(startMask.cpu().numpy())))
            #     if idx0.size == 1:
            #         idx0 = np.resize(idx0, (1,))
            #     m1.weight.data = m0.weight.data[:, idx0].clone()
    print(f"15.newModel = {newModel}")
    return newModel


if __name__ == '__main__':
    main()
