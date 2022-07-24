from sklearn.model_selection import train_test_split, KFold
from glob import glob
import pandas as pd
import pickle
import numpy as np
import random

def MakeDataset(config):
    img_dataset = sorted(glob(config.img_path + '/*.png'))
    mask_dataset = sorted(glob(config.mask_path + '/*.png'))

    img_name = [fName.split("/")[-1].split(".")[0] for fName in img_dataset]
    mask_name = [fName.split("/")[-1].split("_mask")[0] for fName in mask_dataset]
    img_mask = list(set(mask_name).intersection(set(img_name)))\

    img_dataset = [i for i in img_dataset if i.split("/")[-1].split(".")[0] in img_mask]
    mask_dataset = [i for i in mask_dataset if i.split("/")[-1].split("_mask")[0] in img_mask]

    random.seed(config.seed)
    random.shuffle(img_dataset)
    random.seed(config.seed)
    random.shuffle(mask_dataset)

    test_size = int(len(img_dataset)/10)

    test_img = img_dataset[:test_size]
    valid_img = img_dataset[test_size:test_size*2]
    train_img = img_dataset[test_size*2:]

    test_mask = mask_dataset[:test_size]
    valid_mask = mask_dataset[test_size:test_size*2]
    train_mask = mask_dataset[test_size*2:]

    keys = ['train', 'valid', 'test']
    img_list = {key: [] for key in keys}
    mask_list = {key: [] for key in keys}

    img_list['train'] = train_img
    img_list['valid'] = valid_img
    img_list['test'] = test_img

    mask_list['train'] = train_mask
    mask_list['valid'] = valid_mask
    mask_list['test'] = test_mask

    return img_list, mask_list