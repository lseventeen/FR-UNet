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
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance, remove_files, dual_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics_seed, count_connect_component


class Trainer:
    def __init__(self, mode, model, CFG=None, loss=None,
                 train_loader=None,
                 checkpoint=None,
                 test_loader=None,
                 save_path=None, show=False):
        self.CFG = CFG
        self.show = show
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = loss
        self.model = nn.DataParallel(model.cuda())
        cudnn.benchmark = True
        # train and val
        if mode == "train":
            # OPTIMIZER
            self.optimizer = get_instance(
                torch.optim, "optimizer", CFG, self.model.parameters())
            self.lr_scheduler = get_instance(
                torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)

            # CHECKPOINTS
            start_time = datetime.now().strftime('%y%m%d%H%M')
            self.checkpoint_dir = os.path.join(
                CFG.save_dir, self.CFG['model']['type'], start_time)
            dir_exists(self.checkpoint_dir)

            self.train_logger_save_path = os.path.join(
                self.checkpoint_dir, 'runtime.log')
            logger.add(self.train_logger_save_path)
            logger.info(self.checkpoint_dir)

        # test
        if mode == "test":
            self.model.load_state_dict(checkpoint['state_dict'])
            self.checkpoint_dir = save_path

    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            # RUN TRAIN (AND VAL)
            self._train_epoch(epoch)
            # SAVE CHECKPOINT
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch, save_best=True)

        return self.checkpoint_dir

    def _train_epoch(self, epoch):

        self.model.train()
        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

            # LOSS & OPTIMIZE
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
            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()
            # FOR EVAL and INFO
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *
                    self._metrics_ave().values(), self.batch_time.average,
                    self.data_time.average))

        self.lr_scheduler.step()

    def test(self):

        logger.info(f"###### TEST EVALUATION ######")
        self.model.eval()
        tbar = tqdm(self.test_loader, ncols=50)
        imgs = []
        pres = []
        gts = []

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

                imgs.extend(img)
                gts.extend(gt)
                pres.extend(pre)

                total_loss.update(loss.item())
        tic2 = time.time()
        test_time = tic2 - tic1
        logger.info(f'test time:  {test_time}')
        # keep size to dataset
        if self.CFG["data_set"]["name"] == "DRIVE":
            H, W = 584, 565
        elif self.CFG["data_set"]["name"] == "CHASEDB1":
            H, W = 960, 999
        elif self.CFG["data_set"]["name"] == "DCA1":
            H, W = 300, 300
        elif self.CFG["data_set"]["name"] == "STARE":
            H, W = 605, 700
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

        if self.CFG.DTI is True:
            pre_seed = dual_threshold_iteration(
                pres, self.CFG.threshold, self.CFG.threshold_low, True)
            metrics = get_metrics_seed(pres, pre_seed, gts)
            pre_num, gt_num = count_connect_component(pre_seed, gts)
        else:
            metrics = get_metrics(pres, gts, threshold=self.CFG.threshold)
            pre_num, gt_num = count_connect_component(
                pres, gts, threshold=self.CFG.threshold)

        # LOGGING
        tic3 = time.time()
        metrics_time = tic3 - tic1
        logger.info(f'metrics time:  {metrics_time}')
        logger.info(f'         loss: {total_loss.average}')
        for k, v in metrics.items():
            logger.info(f'         {str(k):15s}: {v}')
        logger.info(f'         pre_num: {pre_num}')
        logger.info(f'         gt_num: {gt_num}')

    def _save_checkpoint(self, epoch, save_best=True):
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

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            logger.info("Saving current best: best_model.pth")
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
