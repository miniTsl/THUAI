import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from criterions import LabelSmoothingCrossEntropyLoss, HingeL2Loss
from autoaugment import CIFAR10Policy
from data_augment import RandomCropPaste
from vit import ViT
from basic_CNN import basic_CNN
from Resnet34 import BasicBlock, ResNet

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)  # 标签平滑，交叉熵损失
        else:
            criterion = nn.CrossEntropyLoss()   # 交叉熵损失 
    elif args.criterion=="mse":
        criterion = nn.MSELoss()
    elif args.criterion== "hinge":
        criterion = nn.HingeEmbeddingLoss()
    elif args.criterion== "hinge+l2":
        criterion = HingeL2Loss(args.num_classes)
    else:
        raise ValueError(f"{args.criterion}?")  # 报错
    return criterion

def get_model(args):
    if args.model_name == 'vit':    # Vision Transformer
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )   # Vision Transformer
    elif args.model_name == 'basic_cnn':
        net = basic_CNN(args.in_c, args.num_classes)
    elif args.model_name == 'resnet':
        net = ResNet(BasicBlock, args.in_c, [3, 4, 6, 3], num_classes=args.num_classes)
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")
    return net

def get_transform(args):
    train_transform = []
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.padding)
    ]   # 随机裁剪
    train_transform += [transforms.RandomHorizontalFlip()]  # 随机水平翻转
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())   # CIFAR10数据集的数据增强
        else:
            print(f"No AutoAugment for {args.dataset}")     # 其他数据集没有AutoAugment
    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]   # 转换为张量，归一化
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]    # 随机裁剪、粘贴
    train_transform = transforms.Compose(train_transform)   # 训练集数据增强
    
    
    test_transform = []
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]   # 转换为张量，归一化
    test_transform = transforms.Compose(test_transform)    # 测试集数据增强
    
    return train_transform, test_transform  # 返回训练集和测试集的数据增强

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.in_c = 3   # 输入通道数
        args.num_classes=10  # 类别数
        args.size = 32  # 图片大小
        args.padding = 4    # padding
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]    # 均值和标准差
        train_transform, test_transform = get_transform(args)   # 获取数据增强
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=False)    # 训练集
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=False)   # 测试集

    elif args.dataset == "c100":
        args.in_c = 3  # 输入通道数
        args.num_classes=100    # 类别数
        args.size = 32  # 图片大小
        args.padding = 4    # padding
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]    # 均值和标准差
        train_transform, test_transform = get_transform(args)   # 获取数据增强
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)    # 训练集
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)  # 测试集
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")    # 报错
    
    return train_ds, test_ds    # 返回训练集和测试集

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"  # AutoAugment
    if args.label_smoothing:
        experiment_name+="_ls"  # Label Smoothing
    if args.rcpaste:
        experiment_name+="_rc"  # Random Crop Paste
    if args.cutmix:
        experiment_name+="_cm"  # CutMix
    if args.mixup:
        experiment_name+="_mu"  # MixUp
    if args.off_cls_token:
        experiment_name+="_gap" # GAP
    print(f"Experiment:{experiment_name}")
    return experiment_name
