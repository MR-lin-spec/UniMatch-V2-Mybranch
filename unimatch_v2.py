# -*- coding: utf-8 -*-
"""
UniMatch V2: 半监督语义分割训练脚本
核心功能：基于PyTorch实现半监督语义分割，利用EMA模型生成伪标签，结合有标签/无标签数据训练
支持分布式训练、动态学习率调整、TensorBoard可视化、模型 checkpoint 保存等功能
"""

# ============================== 1. 导入依赖库 ==============================
import argparse  # 命令行参数解析工具，用于接收外部训练配置
from copy import deepcopy  # 深拷贝对象（用于EMA模型初始化，避免浅拷贝导致的参数关联）
import logging  # 日志记录模块，用于输出训练过程信息
import os  # 操作系统交互模块，用于文件路径处理、目录创建等
import pprint  # 格式化打印模块，用于美观输出配置信息

import torch  # PyTorch核心库，提供张量计算和神经网络基础
from torch import nn  # PyTorch神经网络模块，包含常用层和损失函数
import torch.backends.cudnn as cudnn  # CuDNN优化配置，加速GPU训练
from torch.optim import AdamW  # AdamW优化器，缓解权重衰减带来的问题
from torch.utils.data import DataLoader  # 数据加载器，批量加载数据并支持多线程
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化工具，记录训练指标
import yaml  # YAML文件解析器，用于加载训练配置文件

# 自定义模块导入
from dataset.semi import SemiDataset  # 半监督数据集类，处理有标签/无标签数据的加载和增强
from model.semseg.dpt import DPT  # DPT语义分割模型（Encoder-Decoder结构）
from supervised import evaluate  # 模型评估函数，计算mIoU等指标
from util.classes import CLASSES  # 数据集类别定义（如Cityscapes的34类）
from util.ohem import ProbOhemCrossEntropy2d  # OHEM损失函数，解决类别不平衡问题
from util.utils import count_params, init_log, AverageMeter  # 工具函数：参数计数、日志初始化、平均指标统计
from util.dist_helper import setup_distributed  # 分布式训练配置工具，处理多GPU通信

# ============================== 2. 命令行参数解析 ==============================
# 创建参数解析器，描述脚本功能
parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
# 必选参数：训练配置文件路径（包含模型、数据、训练超参数等）
parser.add_argument('--config', type=str, required=True)
# 必选参数：有标签数据的ID列表路径（记录哪些样本是有标签的）
parser.add_argument('--labeled-id-path', type=str, required=True)
# 必选参数：无标签数据的ID列表路径（记录哪些样本是无标签的）
parser.add_argument('--unlabeled-id-path', type=str, required=True)
# 必选参数：模型和日志的保存路径
parser.add_argument('--save-path', type=str, required=True)
# 分布式训练参数：本地进程排名（多GPU训练时自动分配）
parser.add_argument('--local-rank', '--local-rank', default=0, type=int)
# 分布式训练参数：通信端口（可选，默认自动分配）
parser.add_argument('--port', default=None, type=int)

# ============================== 3. 主函数入口 ==============================
def main():
    # 解析命令行参数，获取外部输入的配置
    args = parser.parse_args()

    # ============================== 4. 初始化配置与日志 ==============================
    # 加载YAML配置文件，合并命令行参数和配置文件参数
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # 初始化全局日志（INFO级别，输出训练过程）
    logger = init_log('global', logging.INFO)
    logger.propagate = 0  # 禁止日志向下传播（避免重复输出）

    # 配置分布式训练：获取当前进程排名（rank）和总进程数（world_size）
    rank, world_size = setup_distributed(port=args.port)

    # 仅主进程（rank=0）执行日志输出、目录创建、TensorBoard初始化等操作
    if rank == 0:
        # 合并配置文件和命令行参数，生成完整配置字典
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        # 格式化打印所有配置信息（便于调试和记录）
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        # 初始化TensorBoard写入器，保存训练日志（用于可视化）
        writer = SummaryWriter(args.save_path)
        # 创建模型保存目录（如果不存在）
        os.makedirs(args.save_path, exist_ok=True)

    # ============================== 5. CuDNN配置 ==============================
    cudnn.enabled = True  # 启用CuDNN加速（GPU训练必备）
    cudnn.benchmark = True  # 启用基准模式：自动寻找最优卷积算法（固定输入尺寸时加速效果明显）

    # ============================== 6. 模型初始化 ==============================
    # DPT模型配置字典：不同backbone规模（small/base/large/giant）的参数配置
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    # 解析backbone名称（如'dpt_base' -> 'base'），初始化DPT模型
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    # 加载预训练权重（仅加载backbone部分，解码器随机初始化）
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    
    # 如果配置中要求锁定backbone，设置其参数不参与梯度更新
    if cfg['lock_backbone']:
        model.lock_backbone()

    # ============================== 7. 优化器配置 ==============================
    optimizer = AdamW(
        [
            # 第一组参数：backbone的可训练参数（单独设置学习率）
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            # 第二组参数：除backbone外的其他参数（学习率乘以倍率，如解码器学习率更高）
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'],  # 基础学习率（与第一组参数一致）
        betas=(0.9, 0.999),  # AdamW的动量参数
        weight_decay=0.01  # 权重衰减（正则化，防止过拟合）
    )

    # ============================== 8. 日志输出模型参数信息 ==============================
    if rank == 0:
        # 输出总参数数量、encoder（backbone）参数数量、decoder（head）参数数量
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))

    # ============================== 9. 分布式训练配置 ==============================
    # 获取本地进程排名（用于多GPU分配）
    local_rank = int(os.environ["LOCAL_RANK"])
    # 转换为同步BatchNorm：多GPU训练时，所有GPU的BN统计信息同步（保证训练一致性）
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()  # 将模型移至GPU（默认使用当前进程对应的GPU）

    # 包装为分布式数据并行（DDP）模型：多GPU并行训练核心
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],  # 当前进程使用的GPU ID
        broadcast_buffers=False,  # 禁用缓冲区广播（减少通信开销）
        output_device=local_rank,  # 输出张量的设备
        find_unused_parameters=True, # 允许存在未使用的参数（如锁定的backbone部分）
        #bucket_cap_mb=-1  # 禁用梯度桶，每个参数单独通信
    )

    # ============================== 10. EMA模型初始化（指数移动平均） ==============================
    # 深拷贝原始模型作为EMA模型（用于生成伪标签，提高稳定性）
    model_ema = deepcopy(model)
    model_ema.eval()  # EMA模型始终处于评估模式（禁用Dropout、BN使用移动平均）
    # 冻结EMA模型所有参数（不参与梯度更新，仅通过原始模型参数移动平均更新）
    for param in model_ema.parameters():
        param.requires_grad = False

    # ============================== 11. 损失函数配置 ==============================
    # 有标签数据损失函数：根据配置选择CrossEntropy或OHEM
    if cfg['criterion']['name'] == 'CELoss':
        # 标准交叉熵损失（适用于类别平衡数据）
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        # OHEM损失（Online Hard Example Mining，挖掘难例样本，解决类别不平衡）
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        # 未实现的损失函数报错
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    # 无标签数据损失函数：交叉熵损失（reduction='none'保留每个像素的损失值，后续过滤低置信度样本）
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    # ============================== 12. 数据集与数据加载器 ==============================
    # 初始化无标签训练集（'train_u'模式，应用数据增强）
    trainset_u = SemiDataset(
        cfg['dataset'],  # 数据集名称（如'cityscapes'）
        cfg['data_root'],  # 数据根目录
        'train_u',  # 数据集模式（无标签训练集）
        cfg['crop_size'],  # 训练时的裁剪尺寸
        args.unlabeled_id_path  # 无标签数据ID列表路径
    )
    # 初始化有标签训练集（'train_l'模式，样本数与无标签集匹配，保证迭代同步）
    trainset_l = SemiDataset(
        cfg['dataset'],
        cfg['data_root'],
        'train_l',  # 数据集模式（有标签训练集）
        cfg['crop_size'],
        args.labeled_id_path,  # 有标签数据ID列表路径
        nsample=len(trainset_u.ids)  # 样本数与无标签集一致
    )
    # 初始化验证集（'val'模式，无数据增强，用于评估模型性能）
    valset = SemiDataset(
        cfg['dataset'],
        cfg['data_root'],
        'val'  # 数据集模式（验证集）
    )
    
    # 有标签数据加载器（分布式采样器，保证多GPU数据不重复）
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg['batch_size'],  # 每个GPU的批次大小
        pin_memory=True,  # 锁定内存（加速数据传输到GPU）
        num_workers=4,  # 数据加载线程数
        drop_last=True,  # 丢弃最后一个不完整批次（保证批次大小一致）
        sampler=trainsampler_l  # 分布式采样器
    )
    
    # 无标签数据加载器（与有标签加载器配置一致）
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg['batch_size'],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_u
    )
    
    # 验证集数据加载器（批次大小=1，不丢弃最后一个样本）
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,  # 验证时单样本推理（避免显存不足）
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        sampler=valsampler
    )

    # ============================== 13. 训练状态初始化 ==============================
    total_iters = len(trainloader_u) * cfg['epochs']  # 总迭代次数（批次数量 × 总epoch数）
    previous_best, previous_best_ema = 0.0, 0.0  # 记录最佳mIoU（原始模型/EMA模型）
    best_epoch, best_epoch_ema = 0, 0  # 记录最佳mIoU对应的epoch
    epoch = -1  # 当前epoch初始值（用于断点续训）

    # ============================== 14. 加载checkpoint（断点续训） ==============================
    # 检查是否存在最新的checkpoint文件
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        # 加载checkpoint（CPU上加载，避免GPU不匹配）
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])  # 恢复原始模型权重
        model_ema.load_state_dict(checkpoint['model_ema'])  # 恢复EMA模型权重
        optimizer.load_state_dict(checkpoint['optimizer'])  # 恢复优化器状态（学习率、动量等）
        epoch = checkpoint['epoch']  # 恢复当前epoch
        previous_best = checkpoint['previous_best']  # 恢复最佳mIoU记录
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        # 主进程输出断点续训信息
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    # ============================== 15. 训练主循环 ==============================
    # 从当前epoch开始，迭代至总epoch数
    for epoch in range(epoch + 1, cfg['epochs']):
        # 主进程输出当前epoch信息和历史最佳记录
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        # 初始化损失统计器（AverageMeter用于计算平均损失）
        total_loss = AverageMeter()  # 总损失
        total_loss_x = AverageMeter()  # 有标签数据损失（loss_x）
        total_loss_s = AverageMeter()  # 无标签数据损失（loss_s）
        total_mask_ratio = AverageMeter()  # 有效伪标签比例（高置信度区域占比）

        # 设置分布式采样器的epoch：保证每个epoch数据打乱方式一致（多GPU同步）
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        # 合并有标签和无标签数据加载器（同步迭代）
        loader = zip(trainloader_l, trainloader_u)
        
        model.train()  # 设置模型为训练模式（启用Dropout、BN更新）

        # 迭代训练（每次取一组有标签数据和一组无标签数据）
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            # 将数据移至GPU（与当前进程的GPU匹配）
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            # ============================== 生成无标签数据的伪标签 ==============================
            with torch.no_grad():  # 禁用梯度计算（EMA模型仅用于生成伪标签，不更新）
                pred_u_w = model_ema(img_u_w).detach()  # EMA模型对无标签原始图像（weak augment）预测
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]  # 计算每个像素的置信度（最大概率值）
                mask_u_w = pred_u_w.argmax(dim=1)  # 生成伪标签（取最大概率对应的类别）
            
            # ============================== 无标签数据CutMix增强 ==============================
            # 对两个强增强版本（img_u_s1/img_u_s2）应用CutMix（与数据集预处理的CutMix框对应）
            # CutMix：随机裁剪一个样本的区域，替换到另一个样本中，增强数据多样性
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            # ============================== 模型预测 ==============================
            pred_x = model(img_x)  # 有标签数据预测（输出logits）
            # 无标签数据预测：合并两个强增强版本，启用组件Dropout（comp_drop=True），输出后拆分
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            # ============================== 伪标签CutMix（与图像增强对应） ==============================
            # 对伪标签、置信度、忽略掩码应用相同的CutMix，保证标签与图像匹配
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            # ============================== 损失计算 ==============================
            # 1. 有标签数据损失：直接使用真实标签计算损失
            loss_x = criterion_l(pred_x, mask_x)

            # 2. 无标签数据损失：仅使用高置信度（≥conf_thresh）且非忽略区域的伪标签
            # 计算第一个强增强版本的损失
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            # 过滤掩码：高置信度 + 非忽略区域（ignore_mask≠255）
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            # 归一化：除以有效像素数（避免批次间尺度差异）
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            # 计算第二个强增强版本的损失（与第一个一致）
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            # 无标签总损失：两个强增强版本的平均
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            # 最终总损失：有标签损失和无标签损失的平均
            loss = (loss_x + loss_u_s) / 2.0
            
            # ============================== 反向传播与参数更新 ==============================
            optimizer.zero_grad()  # 清空所有参数的梯度（避免梯度累积）
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            # ============================== 损失与指标统计 ==============================
            total_loss.update(loss.item())  # 更新总损失平均值
            total_loss_x.update(loss_x.item())  # 更新有标签损失平均值
            total_loss_s.update(loss_u_s.item())  # 更新无标签损失平均值
            # 计算有效伪标签比例（高置信度且非忽略区域的像素占比）
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            # ============================== 学习率调整（多项式衰减） ==============================
            iters = epoch * len(trainloader_u) + i  # 当前总迭代次数
            # 多项式衰减公式：lr = initial_lr * (1 - iters/total_iters)^0.9
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            # 更新两组参数的学习率
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            # ============================== 更新EMA模型 ==============================
            # EMA系数：随着迭代次数增加，逐渐趋近于0.996（平衡稳定性和适应性）
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            # 遍历原始模型和EMA模型的参数，按EMA系数更新
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            # 更新缓冲区（如BN的running_mean和running_var）
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            # ============================== TensorBoard日志记录 ==============================
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)  # 总损失
                writer.add_scalar('train/loss_x', loss_x.item(), iters)  # 有标签损失
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)  # 无标签损失
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)  # 有效伪标签比例

            # ============================== 训练日志打印 ==============================
            # 每迭代1/8个批次打印一次日志（避免日志过多）
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))

        # ============================== 16. 验证过程 ==============================
        # 选择评估模式：Cityscapes数据集使用滑动窗口（提高大图像推理精度），其他数据集使用原始尺寸
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        # 评估原始模型和EMA模型的性能（计算mIoU和各类别IoU）
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        
        # 主进程输出验证结果
        if rank == 0:
            # 输出每个类别的IoU（原始模型和EMA模型对比）
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            # 输出平均mIoU
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            # 将验证指标写入TensorBoard
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            # 写入每个类别的IoU
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

        # ============================== 17. 保存模型 ==============================
        # 更新最佳mIoU记录（原始模型）
        is_best = mIoU >= previous_best
        previous_best = max(mIoU, previous_best)
        # 更新EMA模型最佳mIoU记录
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        # 更新最佳epoch
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        # 主进程保存checkpoint
        if rank == 0:
            # 构建checkpoint字典（包含模型、优化器、训练状态）
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            # 保存最新模型（用于断点续训）
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            # 如果是当前最佳模型，保存为best.pth
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

# ============================== 18. 程序入口 ==============================
if __name__ == '__main__':
    main()