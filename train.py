from nets.reback import _Reback_v2
from utils.loss import iou_loss2
from utils.data_loader import BasicDataset
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import csv
import ssl
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime
import cv2
import logging
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


# Cancel certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context


class CoreResult:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


def dice_score(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    return intersection / union


def dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def load_weight(model, weight_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weight_path,
                                 map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k[0:7] == 'module.':
            k = k[7:]  # remove 'module.'
        if list(model_dict.keys())[2].find('module.') != -1:
            k = 'module.' + k
        if (k in model_dict.keys()) and (np.shape(model_dict[k]) == np.shape(v)):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    if len(no_load_key) != 0:
        print('no_load_key: ', no_load_key)
        print('model_dict: ', model_dict.keys())
        exit()
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    return model


def main_train(model):
    # 默认参数
    init_learning_rate = 1e-4
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 1e-5
    amp = False
    save_checkpoint = True

    # 数据集子集地址
    # create dataloaders
    loader_args = dict(batch_size=batch_size,
                       num_workers=num_workers,
                       shuffle=True,
                       pin_memory=True,
                       drop_last=True)

    with open(path_txt_train, "r") as f:
        list_id_train = [i.replace('\n', '') for i in list(f)]
    with open(path_txt_val, "r") as f:
        list_id_val = [i.replace('\n', '') for i in list(f)]

    train_dataset = BasicDataset(list_id_train, dir_img, dir_mask, img_size)
    train_loader = DataLoader(train_dataset, **loader_args)
    val_dataset = BasicDataset(list_id_val, dir_img, dir_mask, img_size)
    val_loader = DataLoader(val_dataset, **loader_args)

    # Set up the optimizer, the loss, the learning rate scheduler
    optimizer = {
        'adam': optim.Adam(model.parameters(), lr=init_learning_rate, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), lr=init_learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10, verbose=True)

    # 损失函数
    loss_wbl = iou_loss2(batch=False)
    loss_wbce = nn.BCELoss(size_average=True)

    # 控制台日志
    logging.info(
        f'''Starting training:
        Epochs:                     {num_epoch}
        Batch size:                 {batch_size}
        Initial learning rate:      {init_learning_rate}

        Checkpoints:                {save_checkpoint}
        Device:                     {device.type}
        Images size:                {img_size}
        Mixed Precision:            {amp}
        ''')

    # begin training
    epoch_low_lr = 0
    best_val_loss = 1
    best_val_dice = 0
    for epoch in range(num_epoch):
        train_num_batch = len(train_dataset.ids) // batch_size
        val_num_batch = len(val_dataset.ids) // batch_size

        print('\nStart Train')
        model.train()
        train_total_loss = 0
        train_total_dice = 0
        with tqdm(total=int(train_num_batch * batch_size), desc=f'Epoch {epoch + 1}/{num_epoch}', unit='img') as pbar:
            for iteration_train, batch in enumerate(train_loader):
                images = batch['image']
                true_masks = batch['mask']

                with torch.no_grad():
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(
                        device=device, dtype=torch.float32)

                optimizer.zero_grad()

                # 前向传播
                out_1st, out_res, out_2nd = model(images)
                loss_all = loss_wbl(true_masks, out_2nd)  # loss_wbl_gp2

                # 计算损失值
                # i.e. loss_step = loss_wbl_gp1 + loss_wbl_gp2 + loss_wbce_rr
                loss_step = loss_all                                            # loss_wbl_gp2
                # loss_wbl_gp1
                loss_step += loss_wbl(true_masks, out_1st)
                res_lab = torch.abs(
                    torch.add(true_masks, torch.neg(out_2nd)))  # 最终预测与gt的绝对差值
                res_lab = res_lab.detach()
                loss_res = loss_wbce(F.interpolate(out_res, size=(img_size, img_size),
                                                   mode='bilinear'), res_lab)
                loss_step += loss_res                                           # loss_wbce_rr

                # 反向传播
                loss_step.backward()
                optimizer.step()

                # 累计loss和dice
                with torch.no_grad():
                    train_total_loss += loss_all.item()
                    train_total_dice += dice_score(predictive=out_2nd,
                                                   target=true_masks).item()

                pbar.update(batch_size)
                pbar.set_postfix(**{'train_epoch_loss': train_total_loss / (iteration_train + 1),
                                    'train_epoch_dice': train_total_dice / (iteration_train + 1),
                                    'lr': optimizer.state_dict()['param_groups'][0]['lr']})
        print('Finish Train')

        print('Start Validation')
        model.eval()
        val_total_loss = 0
        val_total_dice = 0
        with tqdm(total=int(val_num_batch * batch_size), desc=f'Epoch {epoch + 1}/{num_epoch}', unit='img') as pbar:
            for iteration_val, batch in enumerate(val_loader):
                images = batch['image']
                true_masks = batch['mask']

                with torch.no_grad():
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(
                        device=device, dtype=torch.float32)

                    # 前向传播
                    out_1st, out_res, out_2nd = model(images)

                    # 计算损失值
                    loss_all = loss_wbl(true_masks, out_2nd)  # loss_wbl_gp2

                    val_total_loss += loss_all.item()
                    val_total_dice += dice_score(predictive=out_2nd,
                                                 target=true_masks).item()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'val_epoch_loss': val_total_loss / (iteration_val + 1),
                                    'val_epoch_dice': val_total_dice / (iteration_val + 1)
                                    })

        print('Finish Validation')

        print('Epoch:' + str(epoch + 1) + '/' + str(num_epoch))
        print('Total Loss: %.3f || Val Loss: %.3f '
              % (train_total_loss / train_num_batch,
                 val_total_loss / val_num_batch))
        print('lr: %f' % (optimizer.state_dict()['param_groups'][0]['lr']))

        # 调整学习率
        scheduler.step(val_total_loss / val_num_batch)

        if optimizer.state_dict()['param_groups'][0]['lr'] <= 1e-8:
            epoch_low_lr += 1

        # 保存验证集loss最小的权重
        if (val_total_loss/val_num_batch) <= best_val_loss:
            best_val_loss = val_total_loss/val_num_batch
            print("------------save model with best_val_loss ------------")
            print('best_val_loss: ep%03d-train_loss%.3f-train_dice%.3f-val_loss%.3f-val_dice%.3f'
                  % (epoch + 1,
                     train_total_loss / train_num_batch,
                     train_total_dice / train_num_batch,
                     val_total_loss / val_num_batch,
                     val_total_dice / val_num_batch
                     ))
            Path(dir_logs_train).mkdir(parents=True, exist_ok=True)
            torch.save(
                model.module.state_dict(),
                os.path.join(dir_logs_train, 'best_val_loss.pth')
            )

        # 保存验证集dice最大的权重
        if (val_total_dice/val_num_batch) >= best_val_dice:
            best_val_dice = val_total_dice/val_num_batch
            print("------------save model with best_val_dice------------")
            print('best_val_dice: ep%03d-train_loss%.3f-train_dice%.3f-val_loss%.3f-val_dice%.3f'
                  % (epoch + 1,
                     train_total_loss / train_num_batch,
                     train_total_dice / train_num_batch,
                     val_total_loss / val_num_batch,
                     val_total_dice / val_num_batch
                     ))
            Path(dir_logs_train).mkdir(parents=True, exist_ok=True)
            torch.save(
                model.module.state_dict(),
                os.path.join(dir_logs_train, 'best_val_dice.pth')
            )

        if (save_checkpoint and (epoch + 1) % 1 == 0) or (epoch + 1 == num_epoch) or (epoch_low_lr == 10):
            print("------------save model------------")
            Path(dir_logs_train).mkdir(parents=True, exist_ok=True)
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    dir_logs_train,
                    'ep%03d-train_loss%.3f-train_dice%.3f-val_loss%.3f-val_dice%.3f.pth'
                    % (epoch + 1,
                       train_total_loss / train_num_batch,
                       train_total_dice / train_num_batch,
                       val_total_loss / val_num_batch,
                       val_total_dice / val_num_batch
                       )
                )
            )

        if epoch_low_lr == 10:
            return


def marginal_calculate(epoch, tp, tn, fp, fn):
    eps = 1e-8
    # IoU
    IoU_lesion = tp / max(eps, (tp + fp + fn))
    IoU_bg = tn / max(eps, (tn + fp + fn))
    mIoU = (IoU_lesion + IoU_bg) / 2
    # Dice
    Dice_lesion = (2 * IoU_lesion) / (IoU_lesion + 1)
    Dice_bg = (2 * IoU_bg) / (IoU_bg + 1)
    mDice = (Dice_lesion + Dice_bg) / 2
    # Precision
    Precision_lesion = tp / max(eps, (tp + fp))
    Precision_bg = tn / max(eps, (tn + fn))
    mPrecision = (Precision_lesion + Precision_bg) / 2
    # Recall
    Recall_lesion = tp / max(eps, (tp + fn))
    Recall_bg = tn / max(eps, (tn + fp))
    mRecall = (Recall_lesion + Recall_bg) / 2
    # SE
    SE_lesion = tp / max(eps, (tp + fn))
    # SP
    SP_lesion = tn / max(eps, (tn + fp))
    # ACC
    ACC_lesion = (tp + tn) / max(eps, (tp + tn + fp + fn))

    # save results
    results = {'epoch': epoch,
               'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
               'mIoU': mIoU, 'IoU_bg': IoU_bg, 'IoU_lesion': IoU_lesion,
               'mDice': mDice, 'Dice_bg': Dice_bg, 'Dice_lesion': Dice_lesion,
               'mPrecision': mPrecision, 'Precision_bg': Precision_bg, 'Precision_lesion': Precision_lesion,
               'mRecall': mRecall, 'Recall_bg': Recall_bg, 'Recall_lesion': Recall_lesion,
               'SE_lesion': SE_lesion, 'SP_lesion': SP_lesion, 'ACC_lesion': ACC_lesion}

    return results


def evaluate(model_eval,
             weight_id,
             val_loader,
             test_loader):
    print('\nweight_id: ', weight_id)
    if weight_id.find('-') == -1 or weight_id.find('ep') == -1:
        epoch = weight_id
    else:
        epoch = (weight_id.split('-', 1)[0]).replace('ep', '')
    print('Start Evaluation')

    core_result_dict_vt = {}
    for val_test in ['val', 'test']:
        print('dataset: ', val_test)
        eval_loader = {'val': val_loader, 'test': test_loader}[val_test]

        core_result_dict_pixel = {}
        for pixel_threshold in pixel_threshold_list:
            core_result_dict_pixel[str(pixel_threshold)] = CoreResult()

        with tqdm(total=len(eval_loader.dataset.ids), desc=f'Evaluation', unit='img') as pbar:
            for iteration_eval, batch in enumerate(eval_loader):
                images = batch['image']
                true_masks = batch['mask']

                with torch.no_grad():
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(
                        device=device, dtype=torch.float32)
                    true_masks_flatten = true_masks.flatten()

                    # predict
                    out_1st, out_res, out_2nd = model_eval(images)

                    for pixel_threshold in pixel_threshold_list:
                        # binary
                        outs_bin = (out_2nd > pixel_threshold).int()
                        outs_bin_flatten = outs_bin.flatten()
                        # core calculation
                        core_result_dict_pixel[str(pixel_threshold)].tp += torch.sum(
                            ((outs_bin_flatten == 1).int() + (true_masks_flatten == 1).int()) == 2).item()
                        core_result_dict_pixel[str(pixel_threshold)].fp += torch.sum(
                            ((outs_bin_flatten == 1).int() + (true_masks_flatten == 0).int()) == 2).item()
                        core_result_dict_pixel[str(pixel_threshold)].tn += torch.sum(
                            ((outs_bin_flatten == 0).int() + (true_masks_flatten == 0).int()) == 2).item()
                        core_result_dict_pixel[str(pixel_threshold)].fn += torch.sum(
                            ((outs_bin_flatten == 0).int() + (true_masks_flatten == 1).int()) == 2).item()

                pbar.update(images.shape[0])

        core_result_dict_vt[val_test] = core_result_dict_pixel

    print('Finish Evaluation')

    # marginal calculation
    result_dict_vt = {}
    for val_test in ['val', 'test']:
        result_dict_pixel = {}
        for pixel_threshold in pixel_threshold_list:
            results = marginal_calculate(epoch=epoch,
                                         tp=core_result_dict_vt[val_test][str(
                                             pixel_threshold)].tp,
                                         tn=core_result_dict_vt[val_test][str(
                                             pixel_threshold)].tn,
                                         fp=core_result_dict_vt[val_test][str(
                                             pixel_threshold)].fp,
                                         fn=core_result_dict_vt[val_test][str(pixel_threshold)].fn)
            result_dict_pixel[str(pixel_threshold)] = results
        result_dict_vt[val_test] = result_dict_pixel

    torch.cuda.empty_cache()

    return result_dict_vt


def main_eval(model, dir_run, dir_logs_eval):
    # 数据集子集地址
    # create dataloaders
    loader_args = dict(batch_size=batch_size,
                       num_workers=num_workers,
                       shuffle=False,
                       pin_memory=True,
                       drop_last=False)

    with open(path_txt_val, "r") as f:
        list_id_val = [i.replace('\n', '') for i in list(f)]
    with open(path_txt_test, "r") as f:
        list_id_test = [i.replace('\n', '') for i in list(f)]

    val_dataset = BasicDataset(list_id_val, dir_img, dir_mask, img_size)
    val_loader = DataLoader(val_dataset, **loader_args)
    test_dataset = BasicDataset(list_id_test, dir_img, dir_mask, img_size)
    test_loader = DataLoader(test_dataset, **loader_args)

    list_result_dict_vt = []
    for weight_path in natsorted(glob(str(dir_run) + "/*.pth")):
        weight_id = os.path.splitext(weight_path)[
            0].replace('\\', '/').split('/')[-1]

        # load weight
        model = _Reback_v2()
        model_eval = (load_weight(model, weight_path)).eval()

        # single node multi-GPU cards training with DataParallel
        model_eval = nn.DataParallel(model_eval)
        # model_eval = nn.DataParallel(model_eval, device_ids=[1,3])
        if torch.cuda.is_available():
            cudnn.benchmark = True
            model_eval.cuda()
        else:
            model_eval.to(device=device)

        # eval
        result_dict_vt = evaluate(model_eval,
                                  weight_id,
                                  val_loader,
                                  test_loader)
        list_result_dict_vt.append(result_dict_vt)

    # 输出目录
    cvs_dir = dir_logs_eval / 'results'
    Path(cvs_dir).mkdir(parents=True, exist_ok=True)

    for val_test in ['val', 'test']:
        for pixel_threshold in pixel_threshold_list:
            total_results = []
            for result_dict_vt in list_result_dict_vt:
                total_results.append(
                    result_dict_vt[val_test][str(pixel_threshold)])

            # write total_results in csv
            csv_header = total_results[0].keys()
            csv_name = val_test + '-' + str(int(pixel_threshold*100)) + '.csv'
            csv_path = cvs_dir / csv_name
            with open(str(csv_path), 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()
                writer.writerows(total_results)


if __name__ == '__main__':
    """
    实验参数
    """
    img_size = 1024
    batch_size = 8
    num_epoch = 30
    num_workers = 2*batch_size    # 读取数据的进程数量

    # 数据集目录
    dataset_dir = "/home/nfyyadm/glfdir/project01/dataset/aug_dataset/aug_3/20230412_img-aug_3"

    # 训练项目父目录（根目录）
    project_root_path = "/home/nfyyadm/glfdir/project01/"

    # 实验id
    run_id = '9'
    dataset_id = '20230412_img-aug'

    # 已训练权重保存地址
    # dir_run=
    # dir_run = "/home/nfyyadm/glfdir/project01/RF-net/logs/2023_02_23_01_15_25-20230223-4"
    # dir_run = Path(dir_run) / 'train'

    """
    设置上述参数即可，每次开始实验前，请务必仔细检查。
    下列代码一般无需改动，将自动完成训练与评估，并生成权重与结果。
    """

    # 实验日志（地址与名称）
    if not run_id:
        exit()
    run_date = datetime.date.today().strftime('%Y%m%d')
    run_name = run_date + '-' + run_id
    project_name = 'RF-net'
    project_root_path = Path(project_root_path)         # 训练项目父目录（根目录）
    project_path = project_root_path / project_name     # 根目录 / RFNet（训练项目）

    # 数据集
    dataset_dir = Path(dataset_dir)
    dir_img = dataset_dir / 'img_dir'
    dir_mask = dataset_dir / 'ann_dir'
    path_txt_train = str(dataset_dir / dataset_id / 'train.txt')
    path_txt_val = str(dataset_dir / dataset_id / 'val.txt')
    path_txt_test = str(dataset_dir / dataset_id / 'test.txt')

    # 模型
    model = _Reback_v2()
    # model = nn.DataParallel(model)
    # model_name = 'RFNet_Reback_v2'
    # torch.save(model.module.state_dict(), 'pretrain/' + model_name+'.pth')
    # exit()

    # 载入已训练权重
    # pretrain_weight_path = project_path / 'pretrain' / (model_name+'.pth')
    pretrain_weight_path = r"/home/nfyyadm/glfdir/project01/RF-net/weights/train0412/ep090-train_loss0.257-train_dice0.926-val_loss0.268-val_dice0.849.pth"
    if str(pretrain_weight_path):
        print('pretrain_weight_path: ' + str(pretrain_weight_path))
        model = load_weight(model, pretrain_weight_path)

    # cuda（大数据专用）
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[1,0, 2, 3])
    if torch.cuda.is_available():
        cudnn.benchmark = True
        model.to(device=device)
    else:
        model.to(device=device)

    # logs
    logs_time = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    logs_name = logs_time + '-' + run_name
    dir_logs = Path(project_path) / 'logs' / logs_name
    dir_logs_train = dir_logs / 'train'

    # logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # 训练模型
    main_train(model)
    torch.cuda.empty_cache()

    # 评估模型
    pixel_threshold_list = [0.5, 0.75, 0.9]
    dir_run = dir_logs_train
    # dir_run = ""
    # dir_run = Path(dir_run) / 'train'
    print('dir_run: ', dir_run)
    dir_logs_eval = Path(str(dir_run).replace('train', 'eval', 1))
    main_eval(model, dir_run, dir_logs_eval)

    torch.cuda.empty_cache()
