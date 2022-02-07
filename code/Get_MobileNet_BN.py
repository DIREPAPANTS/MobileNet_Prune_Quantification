import torch
from torch import nn
from model_mobilenetV1 import tiny_detector, MultiBoxLoss
from torchsummary import summary


bn_layers = 25  # BN层的层数
prune_percent = 0.5     # 剪枝率


# 进行预剪枝和早期处理工作
def pre_prune(model):
    print(f"3.test model\n")

    # 统计BN层的通道数
    total = 0   # BN层通道数
    i = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i < bn_layers - 1:
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
            if i < bn_layers - 1:
                i += 1
                size = m.weight.data.shape[0]
                bn_shadow[index:(index+size)] = m.weight.data.abs().clone()
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
            if i < bn_layers - 1:
                i += 1
                weight_copy = m.weight.data.clone()
                mask = weight_copy.abs().gt(thre_number).float()
                remain_channels = torch.sum(mask)




def main():
    print(f"1.main function\n")
    model = tiny_detector(n_classes=3)

    # print(f"2.Model Summary\n")
    # summary(model, (3, 224, 224))
    pre_prune(model)


if __name__ == '__main__':
    main()