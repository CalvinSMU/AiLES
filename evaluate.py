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
import pandas as pd
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



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


def load_weight(model, weight_path, device):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weight_path, map_location=device)
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


def marginal_calculate(dataset, number, tp, tn, fp, fn):
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
    results = {'epoch': dataset,
               'number': number,
               'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
               'mIoU': mIoU, 'IoU_bg': IoU_bg, 'IoU_lesion': IoU_lesion,
               'mDice': mDice, 'Dice_bg': Dice_bg, 'Dice_lesion': Dice_lesion,
               'mPrecision': mPrecision, 'Precision_bg': Precision_bg, 'Precision_lesion': Precision_lesion,
               'mRecall': mRecall, 'Recall_bg': Recall_bg, 'Recall_lesion': Recall_lesion,
               'SE_lesion': SE_lesion, 'SP_lesion': SP_lesion, 'ACC_lesion': ACC_lesion}

    return results


def evaluate(model_eval,
             weight_id,
             eval_loader,
             setname,
             setlist):
    print('\nweight_id: ', weight_id)
    if weight_id.find('-') == -1 or weight_id.find('ep') == -1:
        epoch = weight_id
    else:
        epoch = (weight_id.split('-', 1)[0]).replace('ep', '')
    print('Start Evaluation')

    core_result_dict_vt = {}
    for val_test in ['eval']:
        print('dataset: ', setname)
        eval_loader = {'eval': eval_loader}[val_test]

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
                    print(images.shape)
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
    for val_test in ['eval']:
        result_dict_pixel = {}
        for pixel_threshold in pixel_threshold_list:
            results = marginal_calculate(dataset=setname, number=len(setlist), tp=core_result_dict_vt[val_test][str(pixel_threshold)].tp, tn=core_result_dict_vt[val_test][str(
                pixel_threshold)].tn, fp=core_result_dict_vt[val_test][str(pixel_threshold)].fp, fn=core_result_dict_vt[val_test][str(pixel_threshold)].fn)
            result_dict_pixel[str(pixel_threshold)] = results
        result_dict_vt[val_test] = result_dict_pixel

    torch.cuda.empty_cache()

    return result_dict_vt

def data_pipline():
    df=pd.read_excel(str(dataset_dir / 'category-0322.xlsx'))
    
    setname=[]
    setlist=[]
    patient_id=list(set(df["id"]))
    category=list(set(df["category"]))
    location=list(set(df["location"]))
    # dataset=list(set(df["dataset"]))
    dataset=["test"]
    jpg_list=list(df["jpg_name"])
    
    for data in dataset:
        eval_list=list(df[df["dataset"]==data]["jpg_name"])
        setname.append(data)
        setlist.append(eval_list)
        for loc in location:
            try:
                eval_list=list(df[(df["dataset"]==data) & (df["location"]==loc)]["jpg_name"])
                if len(eval_list)>0:
                    setname.append(data+"-"+loc)
                    setlist.append(eval_list)
                else:
                    print("no data：",data+"-"+loc)
            except:
                continue

        for cat in category:
            try:
                eval_list=list(df[(df["dataset"]==data) & (df["category"]==cat)]["jpg_name"])
                if len(eval_list)>0:
                    setname.append(data+"-"+cat)
                    setlist.append(eval_list)
                else:
                    print("no data：",data+"-"+cat)
            except:
                continue

        for loc in location:
            for cat in category:
                eval_list=list(df[(df["dataset"]==data) & (df["category"]==cat) & (df["location"]==loc)]["jpg_name"])
                if len(eval_list)>0:
                    setname.append(data+"-"+loc+"-"+cat)
                    setlist.append(eval_list)
                else:
                    print("no data：",data+"-"+loc+"-"+cat)

    #针对数据细分
    test_id=list(set(df[df["dataset"]=="test"]["id"]))
    for i in patient_id:
        eval_list=list(df[df["id"]==i]["jpg_name"])
        if i in test_id:
            prefix="test-"
            if len(eval_list)>0:
                setname.append(prefix+str(i))
                setlist.append(eval_list)
    #     else:
    #         prefix="train_val-"
    #     # if len(eval_list)>0:
    #     #     setname.append(prefix+str(i))
    #     #     setlist.append(eval_list)
    #     # else:
    #     #     print("no data：",prefix+str(i))
    #
    #     if prefix=="test-":
    #         for j in jpg_list:
    #             eval_list = list(df[(df["id"] == i) & (df["jpg_name"] == j)]["jpg_name"])
    #             if len(eval_list) > 0:
    #                 setname.append(j)
    #                 setlist.append(eval_list)
    # print(setname)
    return setname,setlist


def main_eval(model, dir_run, dir_logs_eval):
    # create dataloaders
    loader_args = dict(batch_size=batch_size,
                       num_workers=num_workers,
                       shuffle=False,
                       pin_memory=True,
                       drop_last=False)

    setname, setlist = data_pipline()
    for index in range(len(setname)):
        list_id_eval = setlist[index]
        dataset = setname[index]

        # dataloader
        eval_dataset = BasicDataset(list_id_eval, dir_img, dir_mask, img_size)
        eval_loader = DataLoader(eval_dataset, **loader_args)

        list_result_dict_vt = []
        for weight_path in natsorted(glob(str(dir_run) + "/*.pth")):
            weight_id = os.path.splitext(weight_path)[
                0].replace('\\', '/').split('/')[-1]

            # load weight
            model = _Reback_v2()
            model_eval = (load_weight(model, weight_path, device)).eval()

            # single node multi-GPU cards training with DataParallel
            model_eval = nn.DataParallel(model_eval, device_ids=gpu_list)
            if torch.cuda.is_available():
                cudnn.benchmark = True
                model_eval.to(device=device)
            else:
                model_eval.to(device=device)

            # eval
            result_dict_vt = evaluate(model_eval,
                                      weight_id,
                                      eval_loader,
                                      dataset,
                                      list_id_eval)
            list_result_dict_vt.append(result_dict_vt)

        cvs_dir = dir_logs_eval / 'results'
        Path(cvs_dir).mkdir(parents=True, exist_ok=True)

        for val_test in ['eval']:
            for pixel_threshold in pixel_threshold_list:
                total_results = []
                for result_dict_vt in list_result_dict_vt:
                    total_results.append(
                        result_dict_vt[val_test][str(pixel_threshold)])

                # write total_results in csv
                csv_header = total_results[0].keys()
                csv_name = val_test + '-' + \
                    str(int(pixel_threshold*100)) + '.csv'
                csv_path = cvs_dir / csv_name
                with open(str(csv_path), 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_header)
                    writer.writeheader()
                    writer.writerows(total_results)


if __name__ == '__main__':
    """
    hyperparameter
    """
    img_size = 1024
    batch_size = 4
    num_epoch = 115
    num_workers = batch_size

    dataset_dir = "/mnt/f/Project/metastatic_carcinoma/Dataset/Final/Dataset/dataset"

    project_root_path = "/mnt/f/Project/metastatic_carcinoma/Model"

    run_id = '6'
    dataset_id = 'img_0318'

    dir_run = "/mnt/f/Project/metastatic_carcinoma/Model/RF-net/weights/"

    if not run_id:
        exit()
    run_date = datetime.date.today().strftime('%Y%m%d')
    run_name = run_date + '-' + run_id
    project_name = 'RF-net'
    project_root_path = Path(project_root_path)         # 训练项目父目录（根目录）
    project_path = project_root_path / project_name     # 根目录 / RFNet（训练项目）

    dataset_dir = Path(dataset_dir)
    dir_img = dataset_dir / 'img_dir'
    dir_mask = dataset_dir / 'ann_dir'
    path_txt_train = str(dataset_dir / dataset_id / 'train.txt')
    path_txt_val = str(dataset_dir / dataset_id / 'val.txt')
    path_txt_test = str(dataset_dir / dataset_id / 'test.txt')

    model = _Reback_v2()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrain_weight_path = ''
    # pretrain_weight_path = project_path / 'pretrain' / 'RFNet_Reback_v2.pth'
    if str(pretrain_weight_path):
        print('pretrain_weight_path: ' + str(pretrain_weight_path))
        model = load_weight(model, pretrain_weight_path, device)

    gpu_list = [0]
    model = nn.DataParallel(model, device_ids=gpu_list)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        model.to(device=device)
    else:
        model.to(device=device)

    logs_time = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    logs_name = logs_time + '-' + run_name
    dir_logs = Path(project_path) / 'logs' / logs_name
    dir_logs_train = dir_logs / 'train0318'

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    pixel_threshold_list = [0.5, 0.75, 0.9]
    dir_run = Path(dir_run) / 'train0318'
    print('dir_run: ', dir_run)
    dir_logs_eval = Path(str(dir_run).replace('train0318', 'eval0412', 1))
    main_eval(model, dir_run, dir_logs_eval)

    torch.cuda.empty_cache()
