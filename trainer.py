import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta


class Trainer:
    def __init__(self, model, CFG=None, loss=None, train_loader=None, val_loader=None):
        self.CFG = CFG
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.model = nn.DataParallel(model.cuda())
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        self.lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        start_time = datetime.now().strftime('%y%m%d%H%M%S')
        self.checkpoint_dir = os.path.join(
            CFG.save_dir, self.CFG['model']['type'], start_time)
        self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
        dir_exists(self.checkpoint_dir)
        cudnn.benchmark = True

    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:
                results = self._valid_epoch(epoch)
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        wrt_mode = 'train'
        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            if self.CFG.amp is True:
                with torch.cuda.amp.autocast(enabled=True):
                    pre = self.model(img)
                    loss = self.loss(pre, gt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre = self.model(img)
                loss = self.loss(pre, gt)
                loss.backward()
                self.optimizer.step()
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            tic = time.time()
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        predict = self.model(img)
                        loss = self.loss(predict, gt)
                else:
                    predict = self.model(img)
                    loss = self.loss(predict, gt)
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)

        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def test(self, dataset):

        logger.info(f"###### TEST EVALUATION ######")
        self.model.eval()
        tbar = tqdm(self.test_loader, ncols=50)
        imgs = []
        pres = []
        gts = []
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        tic1 = time.time()
        total_loss = AverageMeter()
        self._reset_metrics()
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        pre = self.model(img)
                        loss = self.loss(pre, gt)
                else:
                    pre = self.model(img)
                    loss = self.loss(pre, gt)
                pres.extend(pre)
                imgs.extend(img)
                gts.extend(gt)
                total_loss.update(loss.item())
        tic2 = time.time()
        test_time = tic2 - tic1
        logger.info(f'test time:  {test_time}')
        if dataset == "DRIVE":
            H, W = 584, 565
        elif dataset == "CHASEDB1":
            H, W = 960, 999
        elif dataset == "DCA1":
            H, W = 300, 300
        imgs = torch.cat(imgs, 0)
        gts = torch.cat(gts, 0)
        pres = torch.cat(pres, 0)
        if self.CFG["data_set"]["name"] != "CHUAC":
            imgs = TF.crop(imgs, 0, 0, H, W)
            gts = TF.crop(gts, 0, 0, H, W)
            pres = TF.crop(pres, 0, 0, H, W)

        if self.show == True:
            dir_exists("save_picture")
            remove_files("save_picture")
            n, _, _ = imgs.shape
            for i in range(n):
                predict = torch.sigmoid(pres[i]).cpu().detach().numpy()
                predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                cv2.imwrite(
                    f"save_picture/img{i}.png", np.uint8(imgs[i].cpu().numpy()*255))
                cv2.imwrite(f"save_picture/gt{i}.png",
                            np.uint8(gts[i].cpu().numpy()*255))
                cv2.imwrite(f"save_picture/pre{i}.png", np.uint8(predict*255))
                cv2.imwrite(
                    f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))

        if self.CFG.DTI:
            pre_DTI = double_threshold_iteration(
                pres, self.CFG.threshold, self.CFG.threshold_low, True)
            metrics = get_metrics(pres, gts, predict_b=pre_DTI, )
            if self.CFG.CCC:
                pre_num, gt_num = count_connect_component(pre_DTI, gts)
        else:
            metrics = get_metrics(pres, gts, self.CFG.threshold)
            if self.CFG.CCC:
                pre_num, gt_num = count_connect_component(
                    pres, gts, threshold=self.CFG.threshold)
        tic3 = time.time()
        metrics_time = tic3 - tic1
        logger.info(f'metrics time:  {metrics_time}')
        logger.info(f'         loss: {total_loss.average}')
        for k, v in metrics.items():
            logger.info(f'         {str(k):15s}: {v}')
        if self.CFG.CCC:
            logger.info(f'         pre_num: {pre_num}')
            logger.info(f'         gt_num: {gt_num}')

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }
