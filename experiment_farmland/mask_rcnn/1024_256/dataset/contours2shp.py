import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from pycocotools import mask
from preprocess import GDAL_shp_Data, xy2lonlat
from osgeo import gdal
from shapely.geometry import Polygon
from common import open_json, is_dir, get_imglist
from config import Config

os.environ['PROJ_LIB'] = '/opt/conda/pkgs/proj-6.2.1-haa6030c_0/share/proj'


def mask2contours(img_mask, bi_img2img, shift_ul):
    contours, hierarchy = cv2.findContours(image=img_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for j, contour in enumerate(contours):
        areas.append(cv2.contourArea(contour))
    # 出现area为空的情况
    if len(areas) == 0: return None, None
    max_idx = np.argmax(areas)
    # 最大连通域的边缘点
    max_contour = np.squeeze(contours[max_idx], axis=1)

    # 分割结果在原图的位置
    shift_x = shift_ul[bi_img2img[0]][1]
    shift_y = shift_ul[bi_img2img[0]][2]
    shift_contour = max_contour + np.array([shift_x, shift_y])

    tif_name = shift_ul[bi_img2img[0]][0]
    score = bi_img2img[1]
    result = [shift_contour.tolist(), score]   # [contour, score]
    return tif_name, result


def get_contours(pred_path, shiftul_path, cfg):
    results = {}
    encoded_seg = open_json(pred_path)
    shift_ul = open_json(shiftul_path)

    for es in tqdm(encoded_seg, total=len(encoded_seg)):
        # decode
        if es["score"] > cfg.SCORE_THRED:
            img_mask = mask.decode(es["segmentation"])
            img_mask[img_mask == 1] = 255

            # transform mask to shift contours
            bi_img2img = ["{}.png".format(es["image_id"]), es["score"]]
            tif_name, result = mask2contours(img_mask, bi_img2img, shift_ul)
            if tif_name not in results.keys(): results[tif_name] = []
            results[tif_name].append(result)
    return results


def out_shp(test_orimg_dir, outshp_dir, tifs2contours):
    tif_list = get_imglist(test_orimg_dir)
    for tif in tqdm(tif_list, total=len(tif_list)):
        tifpath = os.path.join(test_orimg_dir, tif)
        shp_path = os.path.join(outshp_dir, "{}.shp".format(tif.split('.')[0]))
        polygon_lonlat_list = []
        score_list = []

        # 1 读取栅格影像数据
        dataset = gdal.Open(tifpath)
        polygon_data = tifs2contours[tif]
        for pg_d in polygon_data:
            polygon = pg_d[0]
            score = pg_d[1]
            # 2 坐标转换
            if len(polygon) >= 3:
                polygon_lonlat = Polygon(xy2lonlat(dataset, np.array(polygon)))
                polygon_lonlat_list.append(polygon_lonlat)
                score_list.append(score)

        # 3 数据输出
        shp_data = GDAL_shp_Data(shp_path)
        shp_data.set_shapefile_data(polygon_lonlat_list, score_list)
    

if __name__ == '__main__':
    start_time = time.time()
    cfg = Config()

    train_orimg_dir = rf"{cfg.ORI_DIR}/train/image"
    test_orimg_dir = rf"{cfg.ORI_DIR}/test/image"
    train_shiftul_path = rf"{cfg.COCO_BASEDIR}annotations/train_shiftul.json"
    test_shiftul_path = rf"{cfg.COCO_BASEDIR}/annotations/test_shiftul.json"
    pred_path = rf"{cfg.RES_BASEDIR}/output/results/seg/mask_rcnn_{cfg.MODE}_results_{cfg.EPOCH}.segm.json"
    outshp_dir = rf"{cfg.RES_BASEDIR}/output/out_shp/{cfg.MODE}/{cfg.EPOCH}"
    is_dir(outshp_dir)

    # test
    if cfg.MODE == 'test-dev':
        results = get_contours(pred_path, test_shiftul_path, cfg)
        out_shp(test_orimg_dir, outshp_dir, results)
    # val
    else:
        results = get_contours(pred_path, train_shiftul_path, cfg)
        out_shp(train_orimg_dir, outshp_dir, results)

    end_time = time.time()
    print("time", end_time-start_time)
