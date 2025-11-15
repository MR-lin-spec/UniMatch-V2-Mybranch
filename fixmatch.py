import argparse  # 用于解析命令行参数
from copy import deepcopy  # 用于深拷贝对象（创建EMA教师模型）
import logging  # 用于日志记录
import os  # 用于文件路径操作
import pprint  # 用于格式化打印参数配置

import torch  # PyTorch核心库
from torch import nn  # 神经网络模块
import torch.backends.cudnn as cudnn  # CuDNN配置（加速GPU训练）
from torch.optim import AdamW  # AdamW优化器
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化工具
import yaml  # 用于解析YAML配置文件

from dataset.semi import SemiDataset  # 自定义半监督数据集类
from model.semseg.dpt import DPT  # DPT语义分割模型（作为学生/教师模型）
from supervised import evaluate  # 评估函数（计算mIoU等指标）
from util.classes import CLASSES  # 数据集类别定义（如城市scapes的类别）
from util.ohem import ProbOhemCrossEntropy2d  # OHEM（在线难例挖掘）损失函数
from util.utils import count_params, init_log, AverageMeter  # 工具函数（参数计数、日志初始化、指标平均）
from util.dist_helper import setup_distributed  # 分布式训练配置工具


# 定义命令行参数
parser = argparse.ArgumentParser(description='Reproduced FixMatch with an EMA Teacher for Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True, help='配置文件路径（YAML格式）')
parser.add_argument('--labeled-id-path', type=str, required=True, help='有标签数据的ID文件路径（记录哪些样本有标签）')
parser.add_argument('--unlabeled-id-path', type=str, required=True, help='无标签数据的ID文件路径（记录哪些样本无标签）')
parser.add_argument('--save-path', type=str, required=True, help='模型权重和日志的保存路径')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int, help='分布式训练中本地进程的ID')
parser.add_argument('--port', default=None, type=int, help='分布式训练通信端口')


def main():
    # 解析命令行参数
    args = parser.parse_args()

    # 加载YAML配置文件
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # 初始化日志（全局日志，等级为INFO）
    logger = init_log('global', logging.INFO)
    logger.propagate = 0  # 禁止日志传播（避免多进程重复打印）

    # 初始化分布式训练环境，获取当前进程rank和总进程数world_size
    rank, world_size = setup_distributed(port=args.port)

    # 仅主进程（rank=0）执行日志记录、TensorBoard写入和文件保存操作
    if rank == 0:
        # 合并配置文件参数和命令行参数
        all_args = {**cfg,** vars(args), 'ngpus': world_size}
        # 打印所有参数配置
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        # 初始化TensorBoard写入器（用于可视化训练过程）
        writer = SummaryWriter(args.save_path)
        
        # 创建保存路径（若不存在则新建）
        os.makedirs(args.save_path, exist_ok=True)

    # 启用CuDNN加速，并设置为benchmark模式（自动寻找最优卷积算法，加速训练）
    cudnn.enabled = True
    cudnn.benchmark = True

    # 定义不同规模的模型配置（根据backbone选择）
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    # 根据配置文件中的backbone选择模型参数，初始化DPT模型（学生模型）
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    
    # 加载预训练权重（仅加载backbone部分）
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    
    # 若配置中要求锁死backbone，则冻结backbone参数（仅训练decoder）
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    # 初始化优化器（AdamW），对backbone和其他部分设置不同学习率
    optimizer = AdamW(
        [
            # backbone中需要更新的参数，使用基础学习率
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            # 非backbone参数（如head），学习率为基础学习率的lr_multi倍
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01  # 权重衰减（L2正则化）
    )
    
    # 主进程打印模型参数数量（总参数、encoder参数、decoder参数）
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    # 获取本地进程的GPU编号（分布式训练）
    local_rank = int(os.environ["LOCAL_RANK"])
    # 将模型的BatchNorm转换为SyncBatchNorm（分布式训练中同步统计量）
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 将模型移动到GPU
    model.cuda()

    # 包装模型为分布式数据并行（DP），支持多GPU训练
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    # 深拷贝学生模型作为教师模型（EMA更新）
    model_ema = deepcopy(model)
    model_ema.eval()  # 教师模型始终处于评估模式（不启用dropout等）
    # 冻结教师模型参数（不参与梯度更新）
    for param in model_ema.parameters():
        param.requires_grad = False
    
    # 定义有监督损失函数（根据配置选择CE或OHEM）
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        # OHEM：在线难例挖掘，专注于难分类样本的损失
        criterion_l = ProbOhemCrossEntropy2d(** cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    # 定义无监督损失函数（不直接求平均，后续根据置信度过滤）
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    
    # 初始化无标签训练集（train_u）
    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    # 初始化有标签训练集（train_l），样本数量与无标签集匹配
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    # 初始化验证集
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    # 为有标签数据集创建分布式采样器（确保多进程数据不重复）
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    # 有标签数据加载器
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    # 为无标签数据集创建分布式采样器
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    # 无标签数据加载器
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    # 验证集分布式采样器
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # 验证集数据加载器（batch_size=1，便于评估）
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    # 计算总迭代次数（用于学习率调度）
    total_iters = len(trainloader_u) * cfg['epochs']
    # 记录最佳mIoU（学生模型和教师模型）
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1  # 初始 epoch 编号
    
    # 加载最近的检查点（若存在），支持断点续训
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])  # 加载学生模型权重
        model_ema.load_state_dict(checkpoint['model_ema'])  # 加载教师模型权重
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
        epoch = checkpoint['epoch']  # 恢复 epoch 编号
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    # 训练主循环
    for epoch in range(epoch + 1, cfg['epochs']):
        # 主进程打印当前 epoch 信息和历史最佳结果
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        # 初始化指标记录器（平均损失、平均有监督损失、平均无监督损失、平均掩码比例）
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        # 设置采样器的 epoch（确保多 epoch 数据打乱）
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        # 合并有标签和无标签数据加载器（按批次同步迭代）
        loader = zip(trainloader_l, trainloader_u)
        
        # 学生模型切换到训练模式
        model.train()

        # 迭代训练批次
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _)) in enumerate(loader):
            # 将数据移动到GPU
            img_x, mask_x = img_x.cuda(), mask_x.cuda()  # 有标签数据和标签
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()  # 无标签数据的弱增强和强增强版本
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()  # 忽略掩码（无效区域）和cutmix区域掩码

            # 教师模型生成伪标签（不计算梯度）
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()  # 教师模型对弱增强无标签数据的预测
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]  # 预测的置信度（最大概率）
                mask_u_w = pred_u_w.argmax(dim=1)  # 伪标签（最大概率对应的类别）
            
            # 对强增强的无标签数据应用cutmix（混合样本增强）
            # 将cutmix区域替换为另一张图片的对应区域
            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s.flip(0)[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
            
            # 分离有标签和无标签数据的批次大小
            num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]
            # 学生模型同时处理有标签数据和强增强无标签数据（拼接后前向传播，再分离结果）
            pred_x, pred_u_s = model(torch.cat((img_x, img_u_s))).split([num_lb, num_ulb])
            
            # 对伪标签、置信度和忽略掩码应用cutmix（与输入数据的cutmix对应）
            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]  # 伪标签的cutmix区域替换
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w.flip(0)[cutmix_box == 1]  # 置信度的cutmix区域替换
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]  # 忽略掩码的cutmix区域替换
            
            # 计算有监督损失（学生模型对有标签数据的预测 vs 真实标签）
            loss_x = criterion_l(pred_x, mask_x)

            # 计算无监督损失（学生模型对强增强无标签数据的预测 vs 教师模型生成的伪标签）
            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            # 仅保留高置信度（>= conf_thresh）且非忽略区域的损失
            loss_u_s = loss_u_s * ((conf_u_w_cutmixed >= cfg['conf_thresh']) & (ignore_mask_cutmixed != 255))
            # 求平均（除以有效区域像素数）
            loss_u_s = loss_u_s.sum() / (ignore_mask_cutmixed != 255).sum().item()
            
            # 总损失：有监督损失和无监督损失的平均
            loss = (loss_x + loss_u_s) / 2.0

            # 分布式训练中同步所有进程（确保梯度计算一致）
            torch.distributed.barrier()

            # 梯度清零、反向传播、参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新损失指标
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            # 计算有效伪标签比例（高置信度且非忽略区域的像素占比）
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            # 计算当前迭代次数（用于学习率调度和EMA更新）
            iters = epoch * len(trainloader_u) + i
            # 多项式学习率衰减（lr = initial_lr * (1 - iters/total_iters)^0.9）
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr  # 更新backbone学习率
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']  # 更新其他部分学习率
            
            # EMA更新系数（随迭代次数增加，逐渐接近0.996）
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            # 更新教师模型参数（指数移动平均）
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            # 更新教师模型的buffer（如BatchNorm的running_mean和running_var）
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            # 主进程记录TensorBoard日志
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            # 每迭代1/8个epoch，主进程打印训练状态
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
        # 选择评估模式（城市scapes用滑动窗口，其他用原始尺寸）
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        # 评估学生模型和教师模型的mIoU
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        
        # 主进程打印评估结果
        if rank == 0:
            # 打印每个类别的IoU
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            # 打印平均mIoU
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            # 记录TensorBoard评估日志
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)
        
        # 判断当前学生模型是否为最佳
        is_best = mIoU >= previous_best
        
        # 更新最佳指标和对应epoch
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        # 主进程保存检查点
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),  # 学生模型权重
                'model_ema': model_ema.state_dict(),  # 教师模型权重
                'optimizer': optimizer.state_dict(),  # 优化器状态
                'epoch': epoch,  # 当前epoch
                'previous_best': previous_best,  # 最佳mIoU（学生）
                'previous_best_ema': previous_best_ema,  # 最佳mIoU（教师）
                'best_epoch': best_epoch,  # 最佳学生模型的epoch
                'best_epoch_ema': best_epoch_ema  # 最佳教师模型的epoch
            }
            # 保存最新检查点
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            # 若当前为最佳学生模型，保存最佳检查点
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


# 程序入口
if __name__ == '__main__':
    main()