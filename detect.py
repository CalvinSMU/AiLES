import logging
from pathlib import Path

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import datetime
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob

from utils.data_loader_detect import BasicDataset

from utils.loss import iou_loss2
from nets.reback import _Reback_v2


def load_weight(model, weight_path):
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


def save_png(out, sub_dir, mode_mask, img_id, img_mask, i, res=False):
    pr_list = out.float().cpu().numpy().transpose(0, 2, 3, 1).squeeze(3)
    pr = cv2.resize(pr_list[i], dsize=(img_mask.shape[1], img_mask.shape[0]))

    # 融合原图和热图掩码
    if mode_mask == 0:
        pr = (pr/max(pr.flatten())*255).astype(np.uint8)
        blur = cv2.GaussianBlur(pr, (13, 13), 11)
        heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.3, img_mask, 0.7, 0)
        cv2.imwrite(os.path.join(sub_dir, img_id + '.png'), heatmap_img)

    # 融合原图和二值化掩码
    elif mode_mask == 1:
        # img_mask[pr > 0] = (255, 0, 0)
        # cv2.imwrite(os.path.join(sub_dir, img_id + '.png'), img_mask)
        pr = (pr*255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(pr, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 1, img_mask, 0, 0)
        cv2.imwrite(os.path.join(sub_dir, img_id + '.png'), pr)


def save_png_1(out, out_bin, sub_dir, img_id, img_mask, i):
    pr_list = out.float().cpu().numpy().transpose(0, 2, 3, 1).squeeze(3)
    pr_bin_list = out_bin.float().cpu().numpy().transpose(0, 2, 3, 1).squeeze(3)
    pr = cv2.resize(pr_list[i], dsize=(img_mask.shape[1], img_mask.shape[0]))
    pr_bin = cv2.resize(pr_bin_list[i], dsize=(
        img_mask.shape[1], img_mask.shape[0]))

    # 融合原图和二值化约束热图掩码
    pr[pr_bin != 1] = 0
    pr = (pr / max(pr.flatten()) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(pr, (13, 13), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 1, img_mask, 0, 0)
    cv2.imwrite(os.path.join(sub_dir, img_id + '.png'), heatmap_img)


def detect(model_eval, pic_loader):
    print('Start Detection')

    with tqdm(total=len(pic_loader.dataset.ids), desc=f'Evaluation', unit='img') as pbar:
        for iteration_eval, batch in enumerate(pic_loader):
            images = batch['image']

            with torch.no_grad():
                images = images.to(device=device, dtype=torch.float32)

                # predict
                out_1st, out_res, out_2nd = model_eval(images)
                pr_list = out_2nd.float().cpu().numpy().transpose(0, 2, 3, 1).squeeze(3)

                # 保存原始结果
                for i in range(len(pr_list)):
                    img_id = pic_loader.dataset.ids[iteration_eval *
                                                    len(pr_list) + i]
                    img_path = pic_loader.dataset.img_path_list[iteration_eval * len(
                        pr_list) + i]
                    img_mask = cv2.imread(img_path)

                    # sub_dir_out_1st = result_dir / 'out_1st'
                    # Path(sub_dir_out_1st).mkdir(parents=True, exist_ok=True)
                    # save_png(out_1st, sub_dir_out_1st, mode_mask=0,img_id=img_id, img_mask=img_mask, i=i, res=False)

                    # sub_dir_out_res = result_dir / 'out_res'
                    # Path(sub_dir_out_res).mkdir(parents=True, exist_ok=True)
                    # save_png(out_res, sub_dir_out_res, mode_mask=0,img_id=img_id, img_mask=img_mask, i=i, res=True)

                    # sub_dir_out_2nd = result_dir / 'out_2nd'
                    # Path(sub_dir_out_2nd).mkdir(parents=True, exist_ok=True)
                    # save_png(out_2nd, sub_dir_out_2nd, mode_mask=0,img_id=img_id, img_mask=img_mask, i=i, res=False)

                    # 保存二值化结果图和二值化约束概率图
                    for pixel_threshold in pixel_threshold_list:
                        outs_bin = (out_2nd > pixel_threshold).int()

                        # 二值化结果图
                        sub_dir_name = str(int(pixel_threshold * 100)) + '-1'
                        sub_dir_outs_bin = result_dir / sub_dir_name
                        Path(sub_dir_outs_bin).mkdir(
                            parents=True, exist_ok=True)
                        save_png(outs_bin, sub_dir_outs_bin, mode_mask=1,img_id=img_id, img_mask=img_mask, i=i, res=False)

                        # 二值化约束概率图
                        out_1st
                        # sub_dir_name = str(
                        #     int(pixel_threshold * 100)) + '-2-out_1st'
                        # sub_dir_outs_bin_1st = result_dir / sub_dir_name
                        # Path(sub_dir_outs_bin_1st).mkdir(
                        #     parents=True, exist_ok=True)
                        # save_png_1(out_1st, outs_bin, sub_dir_outs_bin_1st,img_id=img_id, img_mask=img_mask, i=i)

                        out_res
                        sub_dir_name = str(
                            int(pixel_threshold * 100)) + '-2-out_res'
                        sub_dir_outs_bin_res = result_dir / sub_dir_name
                        Path(sub_dir_outs_bin_res).mkdir(
                            parents=True, exist_ok=True)
                        save_png_1(out_res, outs_bin, sub_dir_outs_bin_res,img_id=img_id, img_mask=img_mask, i=i)

                        # # out_2nd
                        # sub_dir_name = str(
                        #     int(pixel_threshold * 100)) + '-2-out_2nd'
                        # sub_dir_outs_bin_2nd = result_dir / sub_dir_name
                        # Path(sub_dir_outs_bin_2nd).mkdir(
                        #     parents=True, exist_ok=True)
                        # save_png_1(out_2nd, outs_bin, sub_dir_outs_bin_2nd,img_id=img_id, img_mask=img_mask, i=i)

            pbar.update(images.shape[0])

    print('Finish Detection')


if __name__ == '__main__':
    """
    实验参数
    """
    img_size = 1024
    batch_size = 2
    num_workers =4   # 读取数据的进程数量

    # 待检测图片目录
    pic_dir = r"G:\Project\metastatic_carcinoma\Model\video_detect\image\74"

    # 已训练权重保存地址
    weight_path = r"G:\Project\metastatic_carcinoma\Model\RF_Net\weights\train0318\ep090-train_loss0.257-train_dice0.926-val_loss0.268-val_dice0.849.pth"
    # 检测结果根目录
    result_root_dir = r'G:\Project\metastatic_carcinoma\Model\RF_Net\detect'

    # 像素阈值
    # pixel_threshold_list = [0.5, 0.75, 0.9]
    pixel_threshold_list = [0.75]

    """
    设置上述参数即可，每次开始实验前，请务必仔细检查。
    下列代码一般无需改动，将自动完成检测，并生成检测结果。
    """
    weight_id = os.path.splitext(weight_path)[
        0].replace('\\', '/').split('/')[-1]
    print('\nweight_id: ', weight_id)

    # 结果输出目录
    result_dir = os.path.join(result_root_dir,
                              datetime.datetime.strftime(datetime.datetime.now(),
                                                         '%Y_%m_%d_%H_%M_%S') + '-' + weight_id)
    result_dir = Path(result_dir)
    print('result_dir: ', result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _Reback_v2()

    # 载入模型
    model_eval = (load_weight(model, weight_path)).eval()

    # cuda（大数据限定）
    gpu_list = [0]
    model_eval = nn.DataParallel(model_eval, device_ids=gpu_list)
    if torch.cuda.is_available():
        cudnn.benchmark = True
    model_eval.to(device=device)

    # logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # dataloader
    loader_args = dict(batch_size=batch_size,
                       num_workers=num_workers,
                       shuffle=False,
                       pin_memory=True,
                       drop_last=False)
    pic_dataset = BasicDataset(pic_dir, img_size)
    pic_loader = DataLoader(pic_dataset, **loader_args)

    # detect
    detect(model_eval, pic_loader)

    torch.cuda.empty_cache()
