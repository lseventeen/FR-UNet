import os
import pickle
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
    
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_pickle(path, type):
    with open(file=path + f"/{type}.pkl", mode='rb') as file:
        img = pickle.load(file)
    return img


def save_pickle(path, type, img_list):
    with open(file=path + f"/{type}.pkl", mode='wb') as file:
        pickle.dump(img_list, file)


def dual_threshold_iteration(list, h_thresh, l_thresh, save=True):
    bin_list = []
    for index, img in enumerate(list):
        img = np.array(torch.sigmoid(img).cpu().detach()*255, dtype=np.uint8)
        bin = np.where(img >= h_thresh*255, 255, 0).astype(np.uint8)
        gbin = bin.copy()
        h, w = img.shape
        gbin_pre = gbin-1

        while(gbin_pre.all() != gbin.all()):
            gbin_pre = gbin
            for i in range(h):
                for j in range(w):
                    if gbin[i][j] == 0 and img[i][j] < h_thresh*255 and img[i][j] >= l_thresh*255:
                        if gbin[i-1][j-1] or gbin[i-1][j] or gbin[i-1][j+1] or gbin[i][j-1] or gbin[i][j+1] or gbin[i+1][j-1] or gbin[i+1][j] or gbin[i+1][j+1]:
                            gbin[i][j] = 255

        if save == True:
            cv2.imwrite(f"save_picture/bin{index}.png", bin)
            cv2.imwrite(f"save_picture/gbin{index}.png", gbin)
        bin_list.append(gbin/255)
    return np.array(bin_list)


def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))