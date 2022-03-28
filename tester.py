import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta


def test(model, Loss, CFG, checkpoint, test_loader, dataset_path, show=False):
    model = nn.DataParallel(model.cuda())
    cudnn.benchmark = True
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"###### TEST EVALUATION ######")
    model.eval()
    tbar = tqdm(test_loader, ncols=50)
    imgs = []
    pres = []
    gts = []
    if CFG.tta:
        model = tta.SegmentationTTAWrapper(
            model, tta.aliases.d4_transform(), merge_mode='mean')
    tic1 = time.time()
    total_loss = AverageMeter()
    with torch.no_grad():
        for img, gt in tbar:
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            pre = model(img)
            loss = Loss(pre, gt)
            pres.extend(pre)
            imgs.extend(img)
            gts.extend(gt)
            total_loss.update(loss.item())
    tic2 = time.time()
    test_time = tic2 - tic1
    logger.info(f'test time:  {test_time}')
    if dataset_path.endswith("DRIVE"):
        H, W = 584, 565
    elif dataset_path.endswith("CHASEDB1"):
        H, W = 960, 999
    elif dataset_path.endswith("DCA1"):
        H, W = 300, 300
    imgs = torch.cat(imgs, 0)
    gts = torch.cat(gts, 0)
    pres = torch.cat(pres, 0)
    if not dataset_path.endswith("CHUAC"):
        imgs = TF.crop(imgs, 0, 0, H, W)
        gts = TF.crop(gts, 0, 0, H, W)
        pres = TF.crop(pres, 0, 0, H, W)

    if show == True:
        dir_exists("save_picture")
        remove_files("save_picture")
        n, _, _ = imgs.shape
        for i in range(n):
            predict = torch.sigmoid(pres[i]).cpu().detach().numpy()
            predict_b = np.where(predict >= CFG.threshold, 1, 0)
            cv2.imwrite(
                f"save_picture/img{i}.png", np.uint8(imgs[i].cpu().numpy()*255))
            cv2.imwrite(f"save_picture/gt{i}.png",
                        np.uint8(gts[i].cpu().numpy()*255))
            cv2.imwrite(f"save_picture/pre{i}.png", np.uint8(predict*255))
            cv2.imwrite(
                f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))

    if CFG.DTI:
        pre_DTI = double_threshold_iteration(
            pres, CFG.threshold, CFG.threshold_low, True)
        metrics = get_metrics(pres, gts, predict_b=pre_DTI)
        if CFG.CCC:
            pre_num, gt_num = count_connect_component(pre_DTI, gts)
    else:
        metrics = get_metrics(pres, gts, CFG.threshold)
        if CFG.CCC:
            pre_num, gt_num = count_connect_component(
                pres, gts, threshold=CFG.threshold)
    tic3 = time.time()
    metrics_time = tic3 - tic1
    logger.info(f'metrics time:  {metrics_time}')
    logger.info(f'         loss: {total_loss.average}')
    for k, v in metrics.items():
        logger.info(f'         {str(k):15s}: {v}')
    if CFG.CCC:
        logger.info(f'         pre_num: {pre_num}')
        logger.info(f'         gt_num: {gt_num}')
