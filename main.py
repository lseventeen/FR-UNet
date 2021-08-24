import argparse
import os
import torch
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
from torchstat import stat


import models
import wandb
from dataset import myDataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch


def train(CFG):
    seed_torch(42)
    train_dataset = myDataset(**CFG["data_set"], mode="training")
    train_loader = DataLoader(
        train_dataset, CFG.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    logger.info('The patch number of train is %d' % len(train_dataset))

    model = get_instance(models, 'model', CFG)
    # stat(model, (1, 48, 48))
    logger.info(f'\n{model}\n')
    # LOSS
    loss = get_instance(losses, 'loss', CFG)
    # TRAINING
    trainer = Trainer(
        model=model,
        mode="train",
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
    )

    return trainer.train()


def test(save_path, checkpoint_name, CFG, show):
    checkpoint = torch.load(os.path.join(save_path, checkpoint_name))
    CFG_ck = checkpoint['config']
    # DATA LOADERS
    test_dataset = myDataset(**CFG["data_set"], mode="test")
    test_loader = DataLoader(test_dataset, batch_size=2,
                             shuffle=False,  num_workers=16, pin_memory=True)
    # MODEL
    model = get_instance(models, 'model', CFG_ck)
    # LOSS
    loss = get_instance(losses, 'loss', CFG_ck)

    # TEST
    tester = Trainer(model=model, mode="test", loss=loss, CFG=CFG, checkpoint=checkpoint,
                     test_loader=test_loader, save_path=save_path, show=show)
    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.yaml', type=str,
                        help='Path to the config file (default: config.yaml)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # yaml = YAML(typ='safe')
    with open('config.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))  # 为列表类型

    os.environ["CUDA_VISIBLE_DEVICES"] = CFG["CUDA_VISIBLE_DEVICES"]
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    path = train(CFG)
    ###############  DRIVE ##################
    # frnet
    # path = "saved/FRNet/2107092336"

    # unet
    # path = "saved/UNet/2102231729"

    # unet++
    # path = "saved/UNet_Nested/2102201102"

    # att-unet
    # path = "saved/AttU_Net/2106260922"

    # CSNET
    # path = "saved/CSNet/2107181027"

    ###############  CHASEDB1 ##################
    # frnet
    # path = "saved/FRNet/2107110954"
    # unet
    # path = "saved/UNet/2102201059"
    
    # unet++
    # path = "saved/UNet_Nested/2102241032"
    # att-unet
    # path = "saved/AttU_Net/2106291024"
    # CSNET
    # path = "saved/CSNet/2107172043"

    ###############  DCA1 ##################
    # frnet
    # path = "saved/FRNet/2107121515"

    # unet
    # path = "saved/UNet/2103292346"
    
    # unet++
    # path = "saved/UNet_Nested/2104100904"
    # att-unet
    # path = "saved/AttU_Net/2106260838"
    # CSNET
    # path = "saved/CSNet/2107172045"

    ###############  CHUAC ##################
    # frnet
    # path = "saved/FRNet/2107122113"

    # unet
    # path = "saved/UNet/2107122120"

    # unet++
    # path = "saved/UNet_Nested/2107131007"
    # att-unet
    # path = "saved/AttU_Net/2107122115"
    # CSNET
    # path = "saved/CSNet/2107182233"
    checkpoint_name = "best_model.pth"
    test(path, checkpoint_name, CFG, show=False)
