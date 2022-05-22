# Copyright (c) 2015-present, Alibaba, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import collections
import logging

# customed libs
from datasets import build_dataset
# from engine import evaluate_vl, train_one_epoch_vl, visual_vl
from losses import DistillationLoss
from samplers import RASampler
from libs import utils, pvlt


def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--config', required=True, type=str, help='config')

    # VL parameters
    parser.add_argument('--num-text-tokens', default=128, type=int, metavar='VL', help='number of text tokens')
    parser.add_argument('--token-hidden-size', default=768, type=int, metavar='VL', help='token hidden size')
    parser.add_argument('--word-mask-rate', default=0.15, type=float, metavar='VL', help='word_mask_rate in masking strategy')
    parser.add_argument('--loss-type', default={'itm':0, 'mlm':0}, type=dict, metavar='VL', help='please indicate the loss type')
    parser.add_argument('--mask-ratio', default=6, type=int, metavar='VL', help='mask ratio in itg task')
    parser.add_argument('--mask-strategy', default='square', type=str, metavar='VL', help='choice: square or stroke or random_grid')
    parser.add_argument('--pretrain-pth', default='/data/oss_bucket_0/PVLT-Data/preweights/pvt_tiny.pth', type=str, metavar='VL', help='please indicate the loss type')
    parser.add_argument('--mask-patch-size', default=16, type=int, metavar='VL', help='choice: square or stroke or random_grid')
    # parser.add_argument('--valid-mask-t2i-loss', default=False, type=bool, metavar='VL', help='choice: square or stroke or random_grid')
    parser.add_argument('--eval-retrieval-itr', action='store_true', help='Perform retrieval_itr only')
    parser.add_argument('--eval-retrieval-tir', action='store_true', help='Perform retrieval_tir only')
    parser.add_argument('--eval-recognition', action='store_true', help='Perform retrieval_tir only')
    
    # Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--viz', action='store_true', help='Perform visualization only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--runtime", default='local', help='env where to run')
    return parser


def main(args):
    # update distributed training parameters
    if args.runtime == "pai":
        print('\n', '*'*40, '\n', '>>> running on PAI <<<\n', '*'*40, '\n')
        utils.init_distributed_mode_on_pai(args)
    elif args.runtime == 'dws':
        print('\n', '*'*40, '\n', '>>> running on DWS <<<\n', '*'*40, '\n')
        utils.init_distributed_mode(args)

    # utils.init_distributed_mode(args)
    print(args)
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")
    
    # 这里可以用于增加不同的mask策略（已删除早期其余尝试，这里不用管）
    if args.mask_strategy == 'random_grid':
        print('>>> using random grid masking strategy for vision! (load from engine_grid_masking.py)')
        from engine_grid_masking import evaluate_vl, train_one_epoch_vl, visual_vl, evaluate_retrieval, evaluate_recognition
    else:
        # print('>>> not using random grid masking strategy! (load from engine.py)')
        # from engine import evaluate_vl, train_one_epoch_vl, visual_vl
        raise Exception('Line-202')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    args.nb_classes = 1000  # 这个参数仅对最后的fully-connected layer作用，所以这里设定不影响自身模型（保留这个参数接口是为了方便后续模型拓展）

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    if True:  # 
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            # Repeated Augmentation (RA)是FAIR在MultiGrain提出的一种抽样策略，一般情况下，训练的mini-batch包含的增强过的sample都是来自不同的图像，但是RA这种抽样策略允许一个mini-batch中包含来自同一个图像的不同增强版本，此时mini-batch的各个样本并非是完全独立的，这相当于对同一个样本进行重复抽样，所以称为Repeated Augmentation。
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    print('>>> TrainBatchSize: {} & ValBatchSize: {}'.format(args.batch_size, int(1.5 * args.batch_size)))
        
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 建立模型
    print(f">>> Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        token_hidden_size=args.token_hidden_size,
        num_text_tokens=args.num_text_tokens,
        loss_type=args.loss_type,
        pretrained_pth=args.pretrain_pth,   # imagenet pvt pretrain
    )
    
    # self.electra_discriminator.apply(fix_bn)    # fix batchnorm [https://www.jb51.net/article/213395.htm]

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f">>> Removing key {k} from pretrained checkpoint")
                logging.info(f">>> Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)
        print('>>> load pretrain weights ({}) of MVLT for downstream finetuning'.format(args.finetune))

    model.to(device)

    model_ema = None

    model_without_ddp = model
    if args.distributed:
        # https://blog.csdn.net/huuuuuuuu/article/details/106381157
        # print('>>> find_unused_parameters')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('>>> number of model parameters:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    # criterion = DistillationLoss(
    #     criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    # )
    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    )
    
    output_dir = Path(args.output_dir)
    # 是否resume模型训练
    if args.resume:
        print('>>> load resume checkpoint from {}'.format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            msg = model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            msg = model_without_ddp.load_state_dict(checkpoint)
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if args.model_ema:
            #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    # 是否处于eval状态
    if args.eval:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print('\n', '*'*40, '\n', '>>> only use evaluation <<<\n', '*'*40, '\n')
        test_stats = evaluate_vl(data_loader_val, model, device, args)
        print(f">>> accuracy of the network on the {len(dataset_val)} test image-text pairs: mlm_acc={test_stats['mlm_acc']:.5f}% itm_acc={test_stats['itm_acc']:.5f}%")
        return
    # 是否处于下游任务的ITR和TIR任务状态
    if args.eval_retrieval_itr or args.eval_retrieval_tir:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print('\n', '*'*40, '\n', '>>> only use eval_retrieval_itr or eval_retrieval_itr <<<\n', '*'*40, '\n')
        test_stats = evaluate_retrieval(data_loader_val, model, device, args)
        # print(f">>> accuracy of the network on the {len(dataset_val)} test image-text pairs: mlm_acc={test_stats['mlm_acc']:.5f}% itm_acc={test_stats['itm_acc']:.5f}%")
        return
    # 是否处于下游任务的父子类识别
    if args.eval_recognition:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=500,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print('\n', '*'*40, '\n', '>>> only use super-sub-class eval_recognition <<<\n', '*'*40, '\n')
        test_stats = evaluate_recognition(data_loader_val, model, device, args)
        # print(f">>> accuracy of the network on the {len(dataset_val)} test image-text pairs: mlm_acc={test_stats['mlm_acc']:.5f}% itm_acc={test_stats['itm_acc']:.5f}%")
        return
    # 是否对网络关键设计进行可视化
    if args.viz:
        print('>>> init data_loader_viz')
        data_loader_viz = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print('\n', '*'*40, '\n', '>>> only use visulization <<<\n', '*'*40, '\n')
        test_stats = visual_vl(data_loader_viz, model, device, args)
        print(f">>> accuracy of the network on the {len(dataset_val)} test image-text pairs: mlm_acc={test_stats['mlm_acc']:.5f}% itm_acc={test_stats['itm_acc']:.5f}%")
        return

    print('>>> init data_loader_train & data_loader_val')
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        )
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    print('\n', '*'*40, '\n', '>>> start training ({} epochs) <<<\n'.format(args.epochs), '*'*40, '\n')

    start_time = time.time()
    total_max_score = 0.0
    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        print('\t---- training at {}/{} epoch ----'.format(epoch, args.epochs))
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_vl(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume, args=args
        )

        lr_scheduler.step(epoch)
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # 前2/3的epoches数量，进行训练
            if epoch < int(args.epochs * 2 // 3):

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
                print('>>> the model is directly saved on {} when [cur_epoch < total_epoch/2]'.format(checkpoint_path))
                cur_itm_accuracy = 0
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
            # 前2/3的epoches数量，使用eval数据进行挑选最佳的epoch
            else:
                # gather the stats from all processes
                test_stats = evaluate_vl(data_loader_val, model, device, args)
                print(f">>> accuracy of the network on the {len(dataset_val)} test image-text pairs:")

                cur_mlm_accuracy = test_stats["mlm_acc"]
                print(f'>>> cur mlm accuracy: {cur_mlm_accuracy:.5f}%')
                cur_itm_accuracy = test_stats["itm_acc"]
                print(f'>>> cur itm accuracy: {cur_itm_accuracy:.5f}%')

                cur_sup_cls_accuracy = test_stats["sup_cls_acc"]
                print(f'>>> cur sup_cls accuracy: {cur_sup_cls_accuracy:.5f}%')
                cur_sub_cls_accuracy = test_stats["sub_cls_acc"]
                print(f'>>> cur sub_cls accuracy: {cur_sub_cls_accuracy:.5f}%')

                total_cur_score = cur_mlm_accuracy + cur_itm_accuracy + cur_sup_cls_accuracy + cur_sub_cls_accuracy

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
                # svae model weights when the following conditions
                if total_cur_score >= total_max_score:    # fix bugs
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)
                    print('>>> successfully find total_cur_score: {} and save model on {}'.format(total_cur_score, checkpoint_path))
                    total_max_score = total_cur_score
                else:
                    print('>>> current score ({}) does not surpass the best one ({}), continue the nex training'.format(total_cur_score, total_max_score))
                    pass

            if args.output_dir and utils.is_main_process():
                with (output_dir / "dws_stdout.log").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('>>> training time {}'.format(total_time_str))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
