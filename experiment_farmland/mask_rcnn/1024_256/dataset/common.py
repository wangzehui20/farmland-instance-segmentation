import json
import os
import numpy as np
from collections import Counter


def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


def seg_double2int(segmentation):
    return [[int(s + 0.5) for seg in segmentation for s in seg]]


def bbox_double2int(bbox):
    return [int(bb + 0.5) for bb in bbox]


def open_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path, result):
    with open(path, 'w') as f:
        f.write(json.dumps(result, indent=1))


def img_filter(file):
    return True if file[-4:] in ['.tif', '.img'] else False


def shp_filter(file):
    return True if file[-4:] in ['.shp'] else False


def get_imglist(tif_dir):
    tif_list_all = os.listdir(tif_dir)
    tif_list = list(filter(img_filter, tif_list_all))
    return tif_list


def get_shplist(shp_dir):
    shp_list_all = os.listdir(shp_dir)
    shp_list = list(filter(shp_filter, shp_list_all))
    return shp_list


def get_mean_std(img):
    means, stdevs = [], []
    img = img[:, :, :, np.newaxis]
    img_band = img.shape[2]
    for i in range(img_band):
        pixels = img[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()
    stdevs.reverse()
    return [means, stdevs]


def is_lowimg(img):
    RMBACKGROUND_THRED = 0.2
    total = img.shape[0] * img.shape[1] * img.shape[2]
    counter = Counter(img.ravel().tolist())
    if counter[0] / total > RMBACKGROUND_THRED:
        return True
    return False
