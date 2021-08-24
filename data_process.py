import argparse
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import safe_load
from scipy import ndimage, stats
from skimage import color, measure
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import removeConnectedComponents
from utils.helpers import dir_exists,remove_files


def data_process(path, name, patch_size, stride, mode, se_size=3, remove_size=200, remove_size2=10):
    data_path = os.path.join(path, name)
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)

    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")

        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":

        file_list = list(sorted(os.listdir(data_path)))
    elif name == "STARE":

        img_path = os.path.join(data_path, mode,"images")
        gt_path = os.path.join(data_path, mode,"gt")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "HRF":

        img_path = os.path.join(data_path, "images")
        gt_path = os.path.join(data_path, "manual1")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "ROSE-1":
        img_path = os.path.join(data_path,"SVC",mode,"img")
        gt_path = os.path.join(data_path,"SVC",mode,"gt")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "ROSE-2":
        img_path = os.path.join(data_path,mode,"original")
        gt_path = os.path.join(data_path,mode,"gt")
        file_list = list(sorted(os.listdir(img_path)))

    img_list = []
    gt_list = []

    for i,file in enumerate(file_list):
        if name == "DRIVE":

            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

        elif name == "CHASEDB1":
            if len(file) == 13:
                if mode == "training" and int(file[6:8]) <= 10:
                    img = Image.open(os.path.join(data_path, file))

                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))

                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(data_path, file))

                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))

        elif name == "HRF":
            

            if mode == "training" and i < 30:
                img = Image.open(os.path.join(img_path, file))

                gt = Image.open(os.path.join(
                        gt_path, file[:-3] + 'tif'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))


            elif mode == "test" and i >= 30:
                img = Image.open(os.path.join(img_path, file))

                gt = Image.open(os.path.join(
                        gt_path, file[:-3] + 'tif'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))

        elif name == "DCA1":
            if len(file) <= 7:

                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))

                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))

        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:

                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                # gt = cv2.resize(
                #     gt, (189, 189), interpolation=cv2.INTER_LINEAR)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)

                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                # gt = cv2.resize(gt, (189, 189), interpolation=cv2.INTER_LINEAR)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
               
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))

        elif name == "STARE":
            
            
        
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:6] + '.vk.ppm'))

            cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
            cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))

            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        
        elif name == "ROSE-1":
            
            
        
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + ".tif"))

            cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
            cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))

            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        elif name == "ROSE-2":
            
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file))

            # cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
            # cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))

            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
                    
        

        # plt.figure(figsize=(12, 36))
        # plt.subplot(311)
        # plt.imshow(img, cmap='gray')
        # plt.subplot(312)
        # plt.imshow(gt, cmap='gray')
        # plt.subplot(313)
        # plt.imshow(mask, cmap='gray')
        # plt.show()
    # print(name)
    # target = torch.cat(gt_list, 0)
    # target = target.cpu().detach().numpy().flatten()
    # count = 0
    # for i in target:
    #     if i != 1 and i != 0:
    #         count += 1
    #         print(f"{count} : {i}")
    # sgt_list, lgt_list = multiscale_gt(gt_list, se_size, remove_size,remove_size2)
    mean, std = getMeanStd(img_list)
    img_list = normalization(img_list, mean, std)
    # cv2.imwrite(f"save_picture/sgt_{name}.png", np.array(sgt_list[0][0])*255)
    # cv2.imwrite(f"save_picture/lgt_{name}.png", np.array(lgt_list[0][0])*255)
    if mode == "training":

        img_patch = get_patch(img_list, patch_size, stride)
        gt_patch = get_patch(gt_list, patch_size, stride)
        # sgt_patch = get_patch(sgt_list, patch_size, stride)
        # lgt_patch = get_patch(lgt_list, patch_size, stride)
        save_image(img_patch, save_path, "img_patch", name)
        save_image(gt_patch, save_path, "gt_patch", name)
        # save_image(sgt_patch, save_path, "sgt_patch", name)
        # save_image(lgt_patch, save_path, "lgt_patch", name)
    elif mode == "test":
        if name != "CHUAC" and name != "ROSE-1" and name != "ROSE-2":
            img_list = get_square(img_list, name)
            gt_list = get_square(gt_list, name)
            # sgt_list = get_square(sgt_list, name)
            # lgt_list = get_square(lgt_list, name)
        save_image(img_list, save_path, "img_square", name)
        save_image(gt_list, save_path, "gt_square", name)
        # save_image(sgt_list, save_path, "sgt_square", name)
        # save_image(lgt_list, save_path, "lgt_square", name)


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list
def multiscale_gt(gt_list, se_size, remove_size, remove_size2):
    gt_small_list = []
    gt_large_list = []
    for gt in gt_list:
        gt = np.squeeze(np.asarray(gt, dtype=np.uint8)) * 255
        ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        gt_large = cv2.morphologyEx(gt, cv2.MORPH_OPEN, ELLIPSE)
        gt_large = removeConnectedComponents(gt_large, remove_size)
        gt_small = gt - gt_large
        gt_small = removeConnectedComponents(gt_small, remove_size2)
        gt_large = torch.from_numpy(gt_large / 255).float()
        gt_small = torch.from_numpy(gt_small / 255).float()

        gt_large_list.append(gt_large.unsqueeze(dim=0))
        gt_small_list.append(gt_small.unsqueeze(dim=0))
    return gt_small_list, gt_large_list

def save_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    elif name == "STARE":
        shape = 704
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s


def getMeanStd(imgs_list):

    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    return mean, std


def normalization(imgs_list, mean, std):
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
        # plt.figure(figsize=(12, 36))
        # plt.subplot(211)
        # plt.imshow(torch.squeeze(i), cmap='gray')
        # plt.subplot(212)
        # plt.imshow(torch.squeeze(n), cmap='gray')
        # plt.show()
    return normal_list





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()


    with open('config.yaml', encoding='utf-8') as file:
        config = safe_load(file)  # 为列表类型

    # data_process(config["data_set"]["path"], name="DRIVE",
    #              mode="training", **config["data_process"])
    # data_process(config["data_set"]["path"], name="DRIVE",
    #              mode="test", **config["data_process"])
    # data_process(config["data_set"]["path"], name="CHASEDB1",
    #              mode="training", **config["data_process"])
    # data_process(config["data_set"]["path"], name="CHASEDB1",
    #              mode="test", **config["data_process"])
    # data_process(config["data_set"]["path"], name="DCA1",
    #              mode="training", **config["data_process"])

    # data_process(config["data_set"]["path"], name="DCA1",
    #              mode="test", **config["data_process"])
    # data_process(config["data_set"]["path"], name="CHUAC",
    #              mode="training", **config["data_process"])

    # data_process(config["data_set"]["path"], name="CHUAC",
    #              mode="test", **config["data_process"])
    # data_process(config["data_set"]["path"], name="STARE",
    #              mode="training", **config["data_process"])

    # data_process(config["data_set"]["path"], name="STARE",
    #              mode="test", **config["data_process"])

    data_process(config["data_set"]["path"], name="ROSE-1",
                 mode="training", **config["data_process"])

    data_process(config["data_set"]["path"], name="ROSE-1",
                 mode="test", **config["data_process"])

    # data_process(config["data_set"]["path"], name="ROSE-2",
    #              mode="training", **config["data_process"])

    # data_process(config["data_set"]["path"], name="ROSE-2",
    #              mode="test", **config["data_process"])