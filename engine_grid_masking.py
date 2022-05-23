"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
# import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, SmoothL1Loss  # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
import os
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from losses import DistillationLoss
from libs import utils
# from evaluation libs
from libs.vl_scores import compute_score_with_logits, compute_mlm_score, compute_psnr
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, precision_score, recall_score

# 设置损失函数的占比（经验值，后续考虑改进？）
MLM_LOSS_WEIGHT, ITM_LOSS_WEIGHT, T2I_LOSS_WEIGHT = 1, 1, 10
USE_ORI_INPUT_IDS = False


def train_one_epoch_vl(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, args=None):
    model.train(set_training_mode)
    # 初始化logging功能
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for idx, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 加载数据样本
        images = samples['image'].to(device, non_blocking=True)   # [bs, 3, 256, 256]
        mlm_labels = samples['mlm_labels'].to(device, non_blocking=True)   # [bs, 128]
        i2t_labels = samples['i2t_labels'].to(device, non_blocking=True)
        masked_images = samples['masked_images'].to(device, non_blocking=True)
        itm_labels = samples['itm_labels'].to(device, non_blocking=True)   # [bs, 1]
        sup_cls_labels = samples['sup_cls_labels'].to(device, non_blocking=True)
        sub_cls_labels = samples['sub_cls_labels'].to(device, non_blocking=True)
        
        # 是否使用原始的input-id
        if USE_ORI_INPUT_IDS:
            # no randomly masking or replacing operations for input_ids
            input_ids = samples['ori_input_ids'].to(device, non_blocking=True)
        else:
            # use randomly masking or replacing operations for input_ids
            input_ids = samples['input_ids'].to(device, non_blocking=True)    # [bs, 128]
        
        # import torchvision.transforms as transforms
        # img = transforms.ToPILImage()(masked_images[0])
        # save_pth = './bak_for_debug_train1207/'
        # os.makedirs(save_pth, exist_ok=True)
        # img.save(save_pth+samples['data_info']['img_name'][0])

        # print('>>> Debug-engine-L93', images.shape, input_ids.shape, mlm_labels.shape, itm_labels.shape)
        if mixup_fn is not None:
            # TODO: seem not be used
            samples, targets = mixup_fn(samples, targets)
        # print('Debug-76:', masked_images.shape, images.shape, n_image.shape)
        with torch.cuda.amp.autocast(enabled=not fp32):
            total_loss = 0
            # 训练 MLM任务和ITM任务
            if idx % 2 == 0:
                outputs = model(images, input_ids)
            # 训练T2I任务
            elif idx % 2 == 1:
                if args.loss_type['t2i'] == 1:
                    # text-to-image
                    outputs = model(masked_images, input_ids)
            
            # calculate loss with different pre-training/down-streaming tasks
            if outputs['mlm_logits'] is not None:
                # print('>>> Debug-mlm_logits: ', outputs['mlm_logits'].shape, outputs['mlm_logits'].view(-1, 30522).shape, mlm_labels.view(-1).shape) 
                # -> [bs, 128, 30522], [16384, 30522], [16384]
                loss_mlm = MLM_LOSS_WEIGHT * CrossEntropyLoss(ignore_index=-1)(outputs['mlm_logits'].view(-1, 30522), mlm_labels.view(-1))

                # print('>>> Debug-MLMhead:', '\n\t', outputs['mlm_logits'].max(), outputs['mlm_logits'].min(), outputs['mlm_logits'].mean(), mlm_labels.max(), mlm_labels.min())
                total_loss += loss_mlm
            
            if outputs['itm_logits'] is not None:
                loss_itm = ITM_LOSS_WEIGHT * CrossEntropyLoss()(outputs['itm_logits'].view(-1, 2), itm_labels.view(-1))
                total_loss += loss_itm
            
            if outputs['sup_cls_logits'] is not None:
                loss_sup_cls = CrossEntropyLoss()(outputs['sup_cls_logits'].view(-1, 48), sup_cls_labels.view(-1))
                loss_sub_cls = CrossEntropyLoss()(outputs['sub_cls_logits'].view(-1, 122), sub_cls_labels.view(-1))
                total_loss += loss_sup_cls
                total_loss += loss_sub_cls
            
            if outputs['t2i_logits'] is not None:

                loss_t2i = T2I_LOSS_WEIGHT * SmoothL1Loss()(outputs['t2i_logits'], images)
                total_loss += loss_t2i

        # @get the scalar value of losses
        total_loss_value = total_loss.item()
        loss_mlm_value = loss_mlm.item() if outputs['mlm_logits'] is not None else 0
        # loss_i2t_value = loss_i2t.item() if outputs['i2t_logits'] is not None else 0
        loss_itm_value = loss_itm.item() if outputs['itm_logits'] is not None else 0
        loss_sup_cls_value = loss_sup_cls.item() if outputs['sup_cls_logits'] is not None else 0
        loss_sub_cls_value = loss_sub_cls.item() if outputs['sub_cls_logits'] is not None else 0
        # loss_itg_value = loss_itg.item() if outputs['itg_logits'] is not None else 0
        loss_t2i_value = loss_t2i.item() if outputs['t2i_logits'] is not None else 0
        # loss_bartMSS_value = loss_bartMSS.item() if outputs['bartMSS_logits'] is not None else 0
        # print(total_loss_value, loss_mlm_value, loss_itm_value)

        if not math.isfinite(total_loss_value):
            # print("Loss is {}, stopping training".format(total_loss_value))
            # sys.exit(1) # replace it with the warning
            print(" [ Warning!!! ] Total Loss is {} (loss_mlm_value={} | loss_i2t_value={} | loss_itm_value={} | loss_sup_cls_value={} | loss_sub_cls_value={} | loss_itg_value={} | loss_t2i_value={} | loss_bartMSS_value={}), raise NaN value".format(
                total_loss_value, loss_mlm_value, loss_i2t_value, loss_itm_value, loss_sup_cls_value, loss_sub_cls_value, loss_itg_value, loss_t2i_value, loss_bartMSS_value))

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(total_loss=total_loss_value)
        metric_logger.update(loss_mlm=loss_mlm_value)
        # metric_logger.update(loss_i2t=loss_i2t_value)
        metric_logger.update(loss_itm=loss_itm_value)
        metric_logger.update(loss_sup_cls=loss_sup_cls_value)
        metric_logger.update(loss_sub_cls=loss_sub_cls_value)
        # metric_logger.update(loss_itg=loss_itg_value)
        metric_logger.update(loss_t2i=loss_t2i_value)
        # metric_logger.update(loss_bartMSS=loss_bartMSS_value)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    del outputs
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_vl(data_loader, model, device, args):
    # criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for _, samples in enumerate(metric_logger.log_every(data_loader, 10, header)):
        bartMSS_input_dict = dict()
        
        images = samples['image'].to(device, non_blocking=True)   # [bs, 3, 256, 256]
        # input_ids = samples['input_ids'].to(device, non_blocking=True)    # [bs, 128]
        # n_images = samples['n_image'].to(device, non_blocking=True)
        masked_images = samples['masked_images'].to(device, non_blocking=True)
        # n_input_ids = samples['n_input_ids'].to(device, non_blocking=True)
        # ori_input_ids = samples['ori_input_ids'].to(device, non_blocking=True) 
        # attention_mask = samples[2].to(device, non_blocking=True)   # [bs, 128]
        mlm_labels = samples['mlm_labels'].to(device, non_blocking=True)   # [bs, 128]
        i2t_labels = samples['i2t_labels'].to(device, non_blocking=True)
        # segment_ids = samples[4].to(device, non_blocking=True)  # [bs, 128]
        itm_labels = samples['itm_labels'].to(device, non_blocking=True)   # [bs, 1]
        sup_cls_labels = samples['sup_cls_labels'].to(device, non_blocking=True)
        sub_cls_labels = samples['sub_cls_labels'].to(device, non_blocking=True)
        bartMSS_input_dict.update(
            input_ids=samples['bartMSS_input_dict']['input_ids'].to(device, non_blocking=True),
            attention_mask=samples['bartMSS_input_dict']['attention_mask'].to(device, non_blocking=True),
            decoder_input_ids=samples['bartMSS_input_dict']['decoder_input_ids'].to(device, non_blocking=True),
            labels=samples['bartMSS_input_dict']['labels'].to(device, non_blocking=True)
            )
        
        input_ids = samples['ori_input_ids'].to(device, non_blocking=True)
        input_ids_mlm = samples['input_ids'].to(device, non_blocking=True)

        # import torchvision.transforms as transforms
        # img = transforms.ToPILImage()(masked_images[0])
        # save_pth = './bak_for_debug_val1207/'
        # os.makedirs(save_pth, exist_ok=True)
        # img.save(save_pth+samples['data_info']['img_name'][0])

        batch_size = images.shape[0]
        # compute output
        with torch.cuda.amp.autocast():
            total_loss = 0
            
            # Part-0
            outputs_mlm = model(images, input_ids_mlm)
            # print(111)
            if outputs_mlm['mlm_logits'] is not None:
                # print('>>> Debug-mlm_logits: ', outputs['mlm_logits'].shape, outputs['mlm_logits'].view(-1, 30522).shape, mlm_labels.view(-1).shape) 
                # -> [bs, 128, 30522], [16384, 30522], [16384]
                loss_mlm = MLM_LOSS_WEIGHT * CrossEntropyLoss(ignore_index=-1)(outputs_mlm['mlm_logits'].view(-1, 30522), mlm_labels.view(-1))
                total_loss += loss_mlm
                # print('>>> Debug-mlm: ', outputs['mlm_logits'].shape, mlm_labels.view(-1).shape)    # torch.Size([240, 128, 30522]) torch.Size([30720])
                # mlm_acc = compute_masked_language_score(outputs['mlm_logits'], mlm_labels.view(-1))
                mlm_acc = compute_mlm_score(outputs_mlm['mlm_logits'], mlm_labels)
                # print(outputs['mlm_logits'], mlm_labels, mlm_acc)
                metric_logger.meters['mlm_acc'].update(mlm_acc, n=batch_size)
            else:
                mlm_acc = 0
                metric_logger.meters['mlm_acc'].update(mlm_acc, n=batch_size)

            # Part-I
            # if args.loss_type['itg'] == 1:
            #     outputs_1 = model(masked_images, input_ids)
            # else:
            #     outputs_1 = model(images, input_ids)

            outputs_1 = model(images, input_ids)
            if outputs_1['itm_logits'] is not None:
                # -> [bs, 1, 2], [bs, 2], [bs, 1]
                # print('>>> Debug-itm_logits: ', itm_labels.min(),itm_labels.max())
                loss_itm = ITM_LOSS_WEIGHT * CrossEntropyLoss()(outputs_1['itm_logits'].view(-1, 2), itm_labels.view(-1))
                total_loss += loss_itm
                # print('>>> Debug-itm: ', outputs['itm_logits'].shape, itm_labels.view(-1).shape)    # torch.Size([240, 1, 2]) torch.Size([240])
                batch_itm_score = compute_score_with_logits(outputs_1['itm_logits'].view(-1, 2), itm_labels.view(-1)).sum()
                itm_acc = batch_itm_score.item() / batch_size
                # print(itm_acc)
                metric_logger.meters['itm_acc'].update(itm_acc, n=batch_size)
            else:
                itm_acc = 0
                metric_logger.meters['itm_acc'].update(itm_acc, n=batch_size)
            
            if outputs_1['sup_cls_logits'] is not None:
                # print('>>> Debug-cls: ', sup_cls_labels.min(),sup_cls_labels.max(), sub_cls_labels.min(),sub_cls_labels.max())
                loss_sup_cls = CrossEntropyLoss()(outputs_1['sup_cls_logits'].view(-1, 48), sup_cls_labels.view(-1))
                loss_sub_cls = CrossEntropyLoss()(outputs_1['sub_cls_logits'].view(-1, 122), sub_cls_labels.view(-1))
                total_loss += loss_sup_cls
                total_loss += loss_sub_cls

                batch_sup_cls_score = compute_score_with_logits(outputs_1['sup_cls_logits'].view(-1, 48), sup_cls_labels.view(-1)).sum()
                batch_sub_cls_score = compute_score_with_logits(outputs_1['sub_cls_logits'].view(-1, 122), sub_cls_labels.view(-1)).sum()
                sup_cls_acc = batch_sup_cls_score.item() / batch_size
                sub_cls_acc = batch_sub_cls_score.item() / batch_size
                
                metric_logger.meters['sup_cls_acc'].update(sup_cls_acc, n=batch_size)
                metric_logger.meters['sub_cls_acc'].update(sub_cls_acc, n=batch_size)
            else:
                sup_cls_acc = 0
                sub_cls_acc = 0
                metric_logger.meters['sup_cls_acc'].update(sup_cls_acc, n=batch_size)
                metric_logger.meters['sub_cls_acc'].update(sub_cls_acc, n=batch_size)
            
            # if outputs_1['itg_logits'] is not None:
            #     # # previous
            #     loss_itg = 10 * SmoothL1Loss()(outputs_1['itg_logits'], images)

            #     # loss_itm = CrossEntropyLoss(ignore_index=-1)(outputs['itm_logits'].view(-1, 2), itm_labels.view(-1))
            #     total_loss += loss_itg
            #     # print('>>> Debug-itm: ', outputs['itm_logits'].shape, itm_labels.view(-1).shape)    # torch.Size([240, 1, 2]) torch.Size([240])
            #     itg_psnr = compute_psnr(outputs_1['itg_logits'], images)
            #     metric_logger.meters['itg_psnr'].update(itg_psnr, n=batch_size)
            # else:
            #     itg_psnr = 0
            #     metric_logger.meters['itg_psnr'].update(itg_psnr, n=batch_size)
            
            # if outputs_1['bartMSS_logits'] is not None:
            #     loss_bartMSS = CrossEntropyLoss(ignore_index=-1)(outputs_1['bartMSS_logits'].view(-1, 30522), bartMSS_input_dict['labels'].view(-1))
            #     total_loss += loss_bartMSS
            #     bartMSS_acc = compute_mlm_score(outputs_1['bartMSS_logits'], bartMSS_input_dict['labels'], index=-1)
            #     metric_logger.meters['bartMSS_acc'].update(bartMSS_acc, n=batch_size)
            # else:
            #     bartMSS_acc = 0
            #     metric_logger.meters['bartMSS_acc'].update(bartMSS_acc, n=batch_size)
            
            # # Part-II
            # if args.loss_type['i2t'] == 1:
            #     # print('Debug-291: load i2t')
            #     outputs_2 = model(images, n_input_ids)    # I2T
            #     if outputs_2['i2t_logits'] is not None:
            #         # print('>>> Debug-mlm_logits: ', outputs['mlm_logits'].shape, outputs['mlm_logits'].view(-1, 30522).shape, mlm_labels.view(-1).shape) 
            #         # -> [bs, 128, 30522], [16384, 30522], [16384]
            #         loss_i2t = 10 * CrossEntropyLoss(ignore_index=-1)(outputs_2['i2t_logits'].view(-1, 30522), i2t_labels.view(-1))
            #         # print('>>> Debug-MLMhead:', '\n\t', outputs['mlm_logits'].max(), outputs['mlm_logits'].min(), outputs['mlm_logits'].mean(), mlm_labels.max(), mlm_labels.min())
            #         total_loss += loss_i2t
            #         # print('Debug-259:', mlm_labels.shape, '\n', mlm_labels, '\n', ori_input_ids.shape, '\n', ori_input_ids)
            #         i2t_acc = compute_mlm_score(outputs_2['i2t_logits'], i2t_labels)
            #         metric_logger.meters['i2t_acc'].update(i2t_acc, n=batch_size)
            #     else:
            #         raise Exception('i2t_logits is none, please check the settings!')
            # else:
            #     i2t_acc = 0
            #     metric_logger.meters['i2t_acc'].update(i2t_acc, n=batch_size)

            # Part-III
            if args.loss_type['t2i'] == 1:
                # print('Debug-291: load t2i')
                # text to image (generation) -> it is optional
                # t2i_input_ids = ori_input_ids.detach()
                # t2i_input_ids[t2i_input_ids[:, :] == -1] = 0
                # outputs_3 = model(n_images, t2i_input_ids)
                outputs_3 = model(masked_images, input_ids)   # T2I
                if outputs_3['t2i_logits'] is not None:
                    # ts = outputs_3['t2i_logits'][0]
                    # ts_norm = (ts - ts.min()) / (ts.max()-ts.min() + 1e-8)
                    # loss_t2i = MSELoss()(ts_norm, images)
                    # print('Debug364', ts_norm.max(), ts_norm.min(), images.max(), images.min())
                    loss_t2i = T2I_LOSS_WEIGHT * SmoothL1Loss()(outputs_3['t2i_logits'], images)
                    total_loss += loss_t2i

                    t2i_psnr = compute_psnr(outputs_3['t2i_logits'], images)
                    metric_logger.meters['t2i_psnr'].update(t2i_psnr, n=batch_size)
                else:
                    raise Exception('t2i_logits is none, please check the settings!')
            else:
                t2i_psnr = 0
                metric_logger.meters['t2i_psnr'].update(t2i_psnr, n=batch_size)

        metric_logger.update(total_loss=total_loss.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('** mlm@acc {mlm_acc.global_avg:.5f} i2t@acc {i2t_acc.global_avg:.5f} itm@acc {itm_acc.global_avg:.5f} sup_cls@acc {sup_cls_acc.global_avg:.5f} sub_cls@acc {sub_cls_acc.global_avg:.5f} itg@psnr {itg_psnr.global_avg:.5f} t2i@psnr {t2i_psnr.global_avg:.5f} bartMSS@acc {bartMSS_acc.global_avg:.5f} loss {total_loss.global_avg:.5f}'
    #       .format(mlm_acc=metric_logger.mlm_acc, i2t_acc=metric_logger.i2t_acc, itm_acc=metric_logger.itm_acc, sup_cls_acc=metric_logger.sup_cls_acc, sub_cls_acc=metric_logger.sub_cls_acc, itg_psnr=metric_logger.itg_psnr, t2i_psnr=metric_logger.t2i_psnr, bartMSS_acc=metric_logger.bartMSS_acc, total_loss=metric_logger.total_loss))
    print('** mlm@acc {mlm_acc.global_avg:.5f} itm@acc {itm_acc.global_avg:.5f} sup_cls@acc {sup_cls_acc.global_avg:.5f} sub_cls@acc {sub_cls_acc.global_avg:.5f} t2i@psnr {t2i_psnr.global_avg:.5f} loss {total_loss.global_avg:.5f}'
          .format(mlm_acc=metric_logger.mlm_acc, itm_acc=metric_logger.itm_acc, sup_cls_acc=metric_logger.sup_cls_acc, sub_cls_acc=metric_logger.sub_cls_acc, t2i_psnr=metric_logger.t2i_psnr, total_loss=metric_logger.total_loss))
    # del outputs

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_retrieval(data_loader, model, device, args):
    save_header = '20220125_retrieval_tir'
    # for ITR & TIR
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    rank_1_count, rank_5_count, rank_10_count = 0, 0, 0 
    for _cur_i, samples in enumerate(metric_logger.log_every(data_loader, 10, header)):
        bartMSS_input_dict = dict()
        
        images = samples['images_101'].to(device, non_blocking=True).squeeze()   # [bs, 3, 256, 256]
        input_ids = samples['ori_input_ids_101'].to(device, non_blocking=True).squeeze()
        info_list = samples['info_list']
        # print('Debug404', images.shape, input_ids.shape)

        batch_size = images.shape[0]
        # compute output
        with torch.cuda.amp.autocast():
            total_loss = 0
            logits = model(images, input_ids)['itm_logits'].view(-1, 2)
            # print('Debug-441', logits.shape, logits)
            logits_softmax = F.softmax(logits, dim=-1)
            # print('Debug-442', logits_softmax.shape, logits_softmax)

            # print(logits_softmax, logits_softmax[:,1])

            sorted_logits, sorted_indices = torch.sort(logits_softmax[:,1], dim=-1, descending=True)
            # print(sorted_logits, sorted_indices)
            
            
            os.makedirs('./visulization/{}/{}/'.format(save_header, '{}.txt'.format(_cur_i)), exist_ok=True)
            tmp_1, tmp_2 = torch.sort(sorted_indices.data, dim=-1, descending=False)
            # print(sorted_indices, type(sorted_indices.data), tmp_1, tmp_2)
            with open('./visulization/{}/{}/ori-text-related.txt'.format(save_header, '{}.txt'.format(_cur_i)), 'w+') as fileobject:
                fileobject.write('>>> >>> info_list <<< <<<\n' + str(info_list) + '\n\n')
                fileobject.write('>>> >>> sorted_logits <<< <<<\n' + str(sorted_logits.data) + '\n\n')
                fileobject.write('>>> >>> sorted_indices <<< <<<\n' + str(sorted_indices.data) + '\n\n')
                for _i in tmp_2.tolist()[0:5]:
                    fileobject.write('>>> >>> rank@5 ({}/5) <<< <<<\n'.format(_i) + str(info_list[_i]) + '\n\n')


            index = np.argwhere(sorted_indices.cpu().numpy() == 0)

            if index < 1: rank_1_count += 1 
            if index < 5: rank_5_count += 1 
            if index < 10: rank_10_count += 1

            # print('[{}/1000] cur_ranking: {} logging: rank@1 ({}), rank@5 ({}), rank@10 ({}) '.format(_cur_i, index, rank_1_count, rank_5_count, rank_10_count))
    
    if args.eval_retrieval_tir:
        flag = 'TIR'
    elif args.eval_retrieval_itr:
        flag = 'ITR'
    print('\n', '#'*30, 'retrieval evaluation', '#'*30 )
    print('>>> retrieval {}: acc@1: {}, acc@5: {}, acc@10: {}'.format(flag, rank_1_count/1000, rank_5_count/1000, rank_10_count/1000))


@torch.no_grad()
def evaluate_recognition(data_loader, model, device, args):
    # for super-category & sub-category recognition
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    save_header = '20220127_recognition'

    # switch to evaluation mode
    model.eval()
    sup_cls_labels_list, sup_cls_preds_list, sub_cls_labels_list, sub_cls_preds_list, name_list = list(), list(), list(), list(), list()
    for _cur_i, samples in enumerate(metric_logger.log_every(data_loader, 10, header)):
        bartMSS_input_dict = dict()
        
        images = samples['images'].to(device, non_blocking=True)   # [bs, 3, 256, 256]
        input_ids = samples['ori_input_ids'].to(device, non_blocking=True)
        sup_cls_labels = samples['sup_cls_labels'].to(device, non_blocking=True)
        sub_cls_labels = samples['sub_cls_labels'].to(device, non_blocking=True)
        info_list = samples['info_list']
        # print(info_list)
        # print('Debug-528', images.shape, input_ids.shape, sup_cls_labels.shape, sub_cls_labels.shape)

        batch_size = images.shape[0]
        # print(batch_size, images.shape)
        # print(type(info_list))

        # compute output
        
        with torch.cuda.amp.autocast():
            total_loss = 0
            logits = model(images, input_ids)

            # print('Debug-496')
            # print(list(sup_cls_labels.view(-1).cpu().numpy()), list(torch.max(F.softmax(logits['sup_cls_logits'].view(-1, 48), dim=-1), dim=-1)[1].cpu().numpy()))
            sup_cls_labels_list += list(sup_cls_labels.view(-1).cpu().numpy())
            sup_cls_preds_list += list(torch.max(F.softmax(logits['sup_cls_logits'].view(-1, 48), dim=-1), dim=-1)[1].cpu().numpy())

            sub_cls_labels_list += list(sub_cls_labels.view(-1).cpu().numpy())
            sub_cls_preds_list += list(torch.max(F.softmax(logits['sub_cls_logits'].view(-1, 122), dim=-1), dim=-1)[1].cpu().numpy())

            name_list += info_list

    # print(len(sup_cls_labels_list), len(sup_cls_preds_list), len(sub_cls_labels_list), len(sub_cls_preds_list))
    # print(sup_cls_labels_list, sup_cls_preds_list, info_list)
    os.makedirs('./visulization/{}/{}/'.format(save_header, '{}.txt'.format(_cur_i)), exist_ok=True)
    with open('./visulization/{}/{}/ori-text-related.txt'.format(save_header, '{}.txt'.format(_cur_i)), 'w+') as fileobject:
        fileobject.write('>>> >>> sup_cls_labels_list <<< <<<\n' + str(sup_cls_labels_list) + '\n\n')
        fileobject.write('>>> >>> sup_cls_preds_list <<< <<<\n' + str(sup_cls_preds_list) + '\n\n')
        fileobject.write('>>> >>> sub_cls_labels_list <<< <<<\n' + str(sub_cls_labels_list) + '\n\n')
        fileobject.write('>>> >>> sub_cls_preds_list <<< <<<\n' + str(sub_cls_preds_list) + '\n\n')
        fileobject.write('>>> >>> name_list <<< <<<\n' + str(name_list) + '\n\n')

        sup_true_list = [sup_cls_labels_list[i] == sup_cls_preds_list[i] for i in range(len(sup_cls_labels_list))]
        sub_true_list = [sub_cls_labels_list[i] == sub_cls_preds_list[i] for i in range(len(sub_cls_labels_list))]
        final_true_list = [((sup_true_list[i] == sub_true_list[i]) and sup_true_list[i] == True and sub_true_list[i] == True) for i in range(len(sub_true_list))]
        final_img_list = [name_list[k] + '@sup_label[{}] @sup_pred[{}] @sub_label[{}] @sub_pred[{}]'.format(sup_cls_labels_list[k],sup_cls_preds_list[k],sub_cls_labels_list[k],sub_cls_preds_list[k]) for k, v in enumerate(final_true_list) if v == True]

        fileobject.write('>>> >>> sup_true_list <<< <<<\n' + str(sup_true_list) + '\n\n')
        fileobject.write('>>> >>> sub_true_list <<< <<<\n' + str(sub_true_list) + '\n\n')
        fileobject.write('>>> >>> final_true_list <<< <<<\n' + str(final_true_list) + '\n\n')
        fileobject.write('>>> >>> final_img_list <<< <<<\n' + str(final_img_list) + '\n\n')

        

    sup_metrics = calculate_cls_metrics(sup_cls_labels_list, sup_cls_preds_list)
    sub_metrics = calculate_cls_metrics(sub_cls_labels_list, sub_cls_preds_list)
    print('\n', '#'*30, 'recognition evaluation', '#'*30 )
    print('> logging-sup: accuracy ({}) macro_f1 ({}) micro_f1 ({}) weighted_f1 ({})\n> logging-sub: accuracy ({}) macro_f1 ({}) micro_f1 ({}) weighted_f1 ({})'.format(sup_metrics[0], sup_metrics[1], sup_metrics[2], sup_metrics[3], sub_metrics[0], sub_metrics[1], sub_metrics[2], sub_metrics[3]))


def calculate_cls_metrics(cls_labels, preds):
    
    # print(cls_labels.shape, preds_softmax_max.shape, '\n', cls_labels, preds_softmax_max)

    micro_f1 = f1_score(cls_labels, preds, average='micro')
    macro_f1 = f1_score(cls_labels, preds, average='macro')
    weighted_f1 = f1_score(cls_labels, preds, average='weighted')
    accuracy = accuracy_score(cls_labels, preds)

    return accuracy, macro_f1, micro_f1, weighted_f1


def tensor2pil(input_tensor, save_pth):
    """only for viz"""
    # print(input_tensor.shape) -> torch.Size([3, 256, 256])
    img = transforms.ToPILImage()(input_tensor)
    # if if_masked:
    #     img_ary = np.array(img)
    #     print(np.unique(img_ary))
    #     img_ary[img_ary == 0] = 128
    #     print(save_pth, img_ary.max(), img_ary.min())
    #     raise Exception
    # res = res.sigmoid().data.cpu().numpy().squeeze()
    # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img.save(save_pth)


def find_valid_logits(logits, target, index):
    """only for viz"""
    logits, target = logits.detach(), target.detach()
    # find the argmax and select the valid value
    preds = logits.argmax(dim=-1)
    valid_preds = preds[target != index]
    valid_target = target[target != index]
    return valid_preds, valid_target


@torch.no_grad()
def visual_vl(data_loader, model, device, args):
    """written by daniel_ji in Mon, 29 Nov"""
    save_header = '20220122-exp21-alibaba'

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for _, samples in enumerate(metric_logger.log_every(data_loader, 10, header)):
        bartMSS_input_dict = dict()
        
        images = samples['image'].to(device, non_blocking=True)   # [bs, 3, 256, 256]
        input_ids = samples['input_ids'].to(device, non_blocking=True)    # [bs, 128]
        n_images = samples['n_image'].to(device, non_blocking=True)
        masked_images = samples['masked_images'].to(device, non_blocking=True)
        masked_images_viz = samples['masked_images'].to(device, non_blocking=True)
        n_input_ids = samples['n_input_ids'].to(device, non_blocking=True)# Daniel
        ori_input_ids = samples['ori_input_ids'].to(device, non_blocking=True) 
        attention_mask = samples[2].to(device, non_blocking=True)   # [bs, 128]
        mlm_labels = samples['mlm_labels'].to(device, non_blocking=True)   # [bs, 128]
        # segment_ids = samples[4].to(device, non_blocking=True)  # [bs, 128]
        itm_labels = samples['itm_labels'].to(device, non_blocking=True)   # [bs, 1]
        sup_cls_labels = samples['sup_cls_labels'].to(device, non_blocking=True)# Daniel
        sub_cls_labels = samples['sub_cls_labels'].to(device, non_blocking=True)# Daniel
        bartMSS_input_dict.update(
            input_ids=samples['bartMSS_input_dict']['input_ids'].to(device, non_blocking=True),
            attention_mask=samples['bartMSS_input_dict']['attention_mask'].to(device, non_blocking=True),
            decoder_input_ids=samples['bartMSS_input_dict']['decoder_input_ids'].to(device, non_blocking=True),
            labels=samples['bartMSS_input_dict']['labels'].to(device, non_blocking=True)
            )
        img_name = samples['data_info']['img_name'][0]
        
        os.makedirs('./visulization/{}/{}/'.format(save_header, img_name), exist_ok=True)
        with open('./visulization/{}/{}/ori-text-related.txt'.format(save_header, img_name), 'w+') as fileobject:
            fileobject.write('>>> >>> input_ids <<< <<<\n' + str(input_ids.data) + '\n\n')
            # fileobject.write('>>> >>> noisy input_ids <<< <<<\n' + str(n_input_ids.data) + '\n\n')
            fileobject.write('>>> >>> original input_ids <<< <<<\n' + str(ori_input_ids.data) + '\n\n')
            fileobject.write('>>> >>> mlm_labels <<< <<<\n' + str(mlm_labels.data) + '\n\n')
            fileobject.write('>>> >>> itm_labels <<< <<<\n' + str(itm_labels.data) + '\n\n')
        
        tensor2pil(input_tensor=images[0], save_pth='./visulization/{}/{}/origin_image.jpg'.format(save_header, img_name))
        tensor2pil(input_tensor=n_images[0], save_pth='./visulization/{}/{}/noise_image.jpg'.format(save_header, img_name))
        masked_images_viz = masked_images_viz[0]
        # print('Debug585', _masked_images.max(), _masked_images.min(), _masked_images.mean())
        masked_images_viz[masked_images_viz == 1e-6] = 0.5
        # print('Debug587', _masked_images.max(), _masked_images.min(), _masked_images.mean())
        # raise Exception
        tensor2pil(input_tensor=masked_images_viz, save_pth='./visulization/{}/{}/masked_image.jpg'.format(save_header, img_name))
        
        batch_size = images.shape[0]
        # compute output
        with torch.cuda.amp.autocast():
            total_loss = 0

            # Part-I
            if args.loss_type['itg'] == 1:
                outputs_1 = model(masked_images, input_ids)
            else:
                outputs_1 = model(images, input_ids)
                
            if outputs_1['mlm_logits'] is not None:
                # print('>>> Debug-mlm_logits: ', outputs['mlm_logits'].shape, outputs['mlm_logits'].view(-1, 30522).shape, mlm_labels.view(-1).shape) 
                # -> [bs, 128, 30522], [16384, 30522], [16384]
                loss_mlm = CrossEntropyLoss(ignore_index=-1)(outputs_1['mlm_logits'].view(-1, 30522), mlm_labels.view(-1))
                total_loss += loss_mlm
                # print('>>> Debug-mlm: ', outputs['mlm_logits'].shape, mlm_labels.view(-1).shape)    # torch.Size([240, 128, 30522]) torch.Size([30720])
                # mlm_acc = compute_masked_language_score(outputs['mlm_logits'], mlm_labels.view(-1))
                mlm_acc = compute_mlm_score(outputs_1['mlm_logits'], mlm_labels)
                # print(outputs['mlm_logits'], mlm_labels, mlm_acc)
                metric_logger.meters['mlm_acc'].update(mlm_acc, n=batch_size)
            else:
                mlm_acc = 0
                metric_logger.meters['mlm_acc'].update(mlm_acc, n=batch_size)

            if outputs_1['itm_logits'] is not None:
                # print('>>> Debug-itm_logits: ', outputs['itm_logits'].shape, outputs['itm_logits'].view(-1, 2).shape, itm_labels.view(-1).shape)
                # -> [bs, 1, 2], [bs, 2], [bs, 1]
                loss_itm = CrossEntropyLoss(ignore_index=-1)(outputs_1['itm_logits'].view(-1, 2), itm_labels.view(-1))
                total_loss += loss_itm
                # print('>>> Debug-itm: ', outputs['itm_logits'].shape, itm_labels.view(-1).shape)    # torch.Size([240, 1, 2]) torch.Size([240])
                batch_itm_score = compute_score_with_logits(outputs_1['itm_logits'].view(-1, 2), itm_labels.view(-1)).sum()
                itm_acc = batch_itm_score.item() / batch_size
                # print(itm_acc)
                metric_logger.meters['itm_acc'].update(itm_acc, n=batch_size)
            else:
                itm_acc = 0
                metric_logger.meters['itm_acc'].update(itm_acc, n=batch_size)
            
            if outputs_1['itg_logits'] is not None:
                loss_itg = 10 * SmoothL1Loss()(outputs_1['itg_logits'], images)
                # loss_itm = CrossEntropyLoss(ignore_index=-1)(outputs['itm_logits'].view(-1, 2), itm_labels.view(-1))
                total_loss += loss_itg
                # print('>>> Debug-itm: ', outputs['itm_logits'].shape, itm_labels.view(-1).shape)    # torch.Size([240, 1, 2]) torch.Size([240])
                itg_psnr = compute_psnr(outputs_1['itg_logits'], images)
                metric_logger.meters['itg_psnr'].update(itg_psnr, n=batch_size)
            else:
                itg_psnr = 0
                metric_logger.meters['itg_psnr'].update(itg_psnr, n=batch_size)
            
            if outputs_1['bartMSS_logits'] is not None:
                loss_bartMSS = CrossEntropyLoss(ignore_index=-1)(outputs_1['bartMSS_logits'].view(-1, 30522), bartMSS_input_dict['labels'].view(-1))
                total_loss += loss_bartMSS
                bartMSS_acc = compute_mlm_score(outputs_1['bartMSS_logits'], bartMSS_input_dict['labels'], index=-1)
                metric_logger.meters['bartMSS_acc'].update(bartMSS_acc, n=batch_size)
            else:
                bartMSS_acc = 0
                metric_logger.meters['bartMSS_acc'].update(bartMSS_acc, n=batch_size)
            
            # Part-II
            if args.loss_type['i2t'] == 1:
                outputs_2 = model(images, n_input_ids)    # I2T
                if outputs_2['i2t_logits'] is not None:
                    # print('>>> Debug-mlm_logits: ', outputs['mlm_logits'].shape, outputs['mlm_logits'].view(-1, 30522).shape, mlm_labels.view(-1).shape) 
                    # -> [bs, 128, 30522], [16384, 30522], [16384]
                    loss_i2t = 10 * CrossEntropyLoss(ignore_index=-1)(outputs_2['i2t_logits'].view(-1, 30522), ori_input_ids.view(-1))
                    # print('>>> Debug-MLMhead:', '\n\t', outputs['mlm_logits'].max(), outputs['mlm_logits'].min(), outputs['mlm_logits'].mean(), mlm_labels.max(), mlm_labels.min())
                    total_loss += loss_i2t
                    # print('Debug-259:', mlm_labels.shape, '\n', mlm_labels, '\n', ori_input_ids.shape, '\n', ori_input_ids)
                    i2t_acc = compute_mlm_score(outputs_2['i2t_logits'], ori_input_ids)
                    metric_logger.meters['i2t_acc'].update(i2t_acc, n=batch_size)
                else:
                    raise Exception('i2t_logits is none, please check the settings!')
            else:
                i2t_acc = 0
                metric_logger.meters['i2t_acc'].update(i2t_acc, n=batch_size)

            # Part-III
            if args.loss_type['t2i'] == 1:
                outputs_3 = model(masked_images, input_ids)   # T2I
                if outputs_3['t2i_logits'] is not None:
                    loss_t2i = 10 * SmoothL1Loss()(outputs_3['t2i_logits'], images)
                    total_loss += loss_t2i

                    t2i_psnr = compute_psnr(outputs_3['t2i_logits'], images)
                    metric_logger.meters['t2i_psnr'].update(t2i_psnr, n=batch_size)
                else:
                    raise Exception('t2i_logits is none, please check the settings!')
            else:
                t2i_psnr = 0
                metric_logger.meters['t2i_psnr'].update(t2i_psnr, n=batch_size)

            # viz functions
            with open('./visulization/{}/{}/output-text-related.txt'.format(save_header, img_name), 'w+') as fileobject:
                # mlm
                if outputs_1['bartMSS_logits'] is not None:
                    valid_preds, valid_target = find_valid_logits(logits=outputs_1['mlm_logits'], target=mlm_labels, index=-1)
                    fileobject.write('>>> >>> MLM task (valid_preds) <<< <<<\n' + str(valid_preds) + '\n')
                    fileobject.write('>>> >>> MLM task (valid_target) <<< <<<\n' + str(valid_target) + '\n\n')
                # itm
                if outputs_1['itm_logits'] is not None:
                    fileobject.write('>>> >>> ITM task (valid_preds) <<< <<<\n' + str(outputs_1['itm_logits'].argmax(dim=-1).data) + '\n')
                    fileobject.write('>>> >>> ITM task (valid_preds) <<< <<<\n' + str(itm_labels.data) + '\n\n')
                # itg
                if outputs_1['itg_logits'] is not None:
                    tensor2pil(input_tensor=outputs_1['itg_logits'][0], save_pth='./visulization/{}/{}/itg_pred.jpg'.format(save_header, img_name))
                    tensor2pil(input_tensor=images[0], save_pth='./visulization/{}/{}/itg_target.jpg'.format(save_header, img_name))
                # i2t
                if outputs_1['i2t_logits'] is not None:
                    valid_preds, valid_target = find_valid_logits(logits=outputs_2['i2t_logits'], target=ori_input_ids, index=-1)
                    fileobject.write('>>> >>> I2T task (valid_preds) <<< <<<\n' + str(valid_preds) + '\n')
                    fileobject.write('>>> >>> I2T task (valid_target) <<< <<<\n' + str(valid_target) + '\n\n')
                # t2i
                if outputs_1['t2i_logits'] is not None:
                    # add norm to ensure the range between [0,1]
                    # print('Debug-544:', outputs_3['t2i_logits'][0].max(), outputs_3['t2i_logits'][0].min(), outputs_3['t2i_logits'][0].mean())
                    ts = outputs_3['t2i_logits'][0]
                    ts_norm = (ts - ts.min()) / (ts.max()-ts.min() + 1e-8)
                    # print('Debug-547:', ts_norm.max(), ts_norm.min(), ts_norm.mean())
                    tensor2pil(input_tensor=ts_norm, save_pth='./visulization/{}/{}/t2i_pred.jpg'.format(save_header, img_name))
                    tensor2pil(input_tensor=images[0], save_pth='./visulization/{}/{}/t2i_target.jpg'.format(save_header, img_name))

        metric_logger.update(total_loss=total_loss.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('** mlm@acc {mlm_acc.global_avg:.5f} i2t@acc {i2t_acc.global_avg:.5f} itm@acc {itm_acc.global_avg:.5f} itg@psnr {itg_psnr.global_avg:.5f} t2i@psnr {t2i_psnr.global_avg:.5f} bartMSS@acc {bartMSS_acc.global_avg:.5f} loss {total_loss.global_avg:.5f}'
          .format(mlm_acc=metric_logger.mlm_acc, i2t_acc=metric_logger.i2t_acc, itm_acc=metric_logger.itm_acc, itg_psnr=metric_logger.itg_psnr, t2i_psnr=metric_logger.t2i_psnr, bartMSS_acc=metric_logger.bartMSS_acc, total_loss=metric_logger.total_loss))

    # del outputs

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}