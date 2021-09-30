import sys
import os.path as osp
import numpy as np
import shapefile
import cv2
from shapely.geometry import Polygon
from preprocess import GDAL_shp_Data, xy2lonlat, lonlat2xy, TifInfo
from osgeo import gdal
from removeshp_overlap import is_union
from tabulate import tabulate
from tqdm import tqdm
from collections import Counter

sys.path.append('..')
from utils.config import Config
from utils.common import open_json, is_dir, get_shplist, get_imglist

cfg = Config()


def get_gt(jsonpath, shiftul_path, orimg_dir, valshp_dir):
    valjson = open_json(jsonpath)
    shift_ul = open_json(shiftul_path)
    polygon_dict = {}
    for seg in valjson['annotations']:
        img_name = "{}.png".format(seg['image_id'])
        orimg_name = shift_ul[img_name][0]
        if orimg_name not in polygon_dict.keys():
            polygon_dict[orimg_name] = []
        for s in seg['segmentation']:
            contours = []
            for i in range(0, len(s), 2):
                contours.append([s[i], s[i+1]])
            shift_contours = np.array(contours) + np.array(shift_ul[img_name][1:3])
        polygon_dict[orimg_name].append(shift_contours)

    polygonshp_dict = {}
    n_gt = 0
    for name, contours in polygon_dict.items():
        orimg_path = osp.join(orimg_dir, name)
        shp_name = "{}.shp".format(name.split('.')[0])
        shp_path = osp.join(valshp_dir, shp_name)
        dataset = gdal.Open(orimg_path)
        polygon_list = []
        for cts in contours:
            lonlats = xy2lonlat(dataset, cts)
            polygon_list.append(Polygon(lonlats))
        
        score_list = [1 for i in range(len(polygon_list))]
        shp_data = GDAL_shp_Data(shp_path)
        shp_data.set_shapefile_data(polygon_list, score_list)
        polygonshp_dict[shp_name] = polygon_list
        n_gt += len(score_list)
    return polygonshp_dict, n_gt


def get_pred(shpdir):
    shp_list = get_shplist(shpdir)
    polygon_dict = {}
    score_dict = {}
    for shp in shp_list:
        shp_path = osp.join(shpdir, shp)
        reader = shapefile.Reader(shp_path)
        polygon_list = []
        score_list = []
        for sr in reader.shapeRecords():
            points = sr.shape.points
            polygon_list.append(Polygon(points))
            score_list.append(sr.record[0])
        polygon_dict[shp] = polygon_list
        score_dict[shp] = score_list
    return polygon_dict, score_dict


def cal_iou(gt, pred, orimg_dir):
    orimg_list = get_imglist(orimg_dir)
    iou = []
    for id, orimg in tqdm(enumerate(orimg_list), total=len(orimg_list)):
        orimg_path = osp.join(orimg_dir, orimg)
        shp = "{}.shp".format(orimg.split('.')[0])
        dataset = gdal.Open(orimg_path)
        img_info = TifInfo(dataset)
        gt_img = np.zeros((img_info.height, img_info.width, 3), dtype=np.uint8)
        pred_img = np.zeros((img_info.height, img_info.width, 3), dtype=np.uint8)

        gt_polygon = gt[shp]
        pred_polygon = pred[shp]
        # fill polygon
        gt_p_xy = []
        pred_p_xy = []
        for gt_p in gt_polygon:
            (gt_p_x, gt_p_y) = polygon_lonlat2xy(gt_p, dataset)
            gt_p_xy.append(np.array([[int(x), int(y)] for x, y in zip(gt_p_x, gt_p_y)]))
            cv2.fillPoly(gt_img, gt_p_xy, (1,1,1))
        for pred_p in pred_polygon:
            (pred_p_x, pred_p_y) = polygon_lonlat2xy(pred_p, dataset)
            pred_p_xy.append(np.array([[int(x), int(y)] for x, y in zip(pred_p_x, pred_p_y)]))
            cv2.fillPoly(pred_img, pred_p_xy, (1,1,1))
        # only consider one channel
        gt_img = gt_img[:,:,0]
        pred_img = pred_img[:,:,0]
        tp = np.sum(gt_img * pred_img)
        fp = np.sum((gt_img==0) * (pred_img==1))
        fn = np.sum((gt_img==1) * (pred_img==0))
        iou.append(tp / (tp + fp + fn))
    return np.mean(iou)


def polygon_lonlat2xy(polygon, dataset):
    points_lonlat = np.array(list(polygon.exterior.coords))
    points_xy = lonlat2xy(dataset, points_lonlat[:, 0], points_lonlat[:, 1])
    return points_xy


def cal_tp(gt, pred):
    tp_dict = {}
    for id, shp in tqdm(enumerate(gt.keys()), total=len(gt.keys())):
        gt_polygon = gt[shp]
        pred_polygon = pred[shp]
        tp_list = np.zeros(len(pred_polygon)).tolist()
        for gt_p in gt_polygon:
            for i, pred_p in enumerate(pred_polygon):
                if is_union([gt_p, pred_p]):
                    gt_p = gt_p.buffer(1e-9)
                    pred_p = pred_p.buffer(1e-9)
                    inter = gt_p.intersection(pred_p)
                    inter_area = inter.area
                    union_area = gt_p.area + pred_p.area - inter_area
                    iou = inter_area / union_area
                    if iou >= 0.5:
                        tp_list[i] = 1
        tp_dict[shp] = tp_list
    return tp_dict


def cal_score(tp_dict, score_dict, iou, n_gt):
    tp_list = []
    score_list = []
    # tp and score are consistent
    for shp in tp_dict.keys():
        tp_list.extend(tp_dict[shp])
        score_list.extend(score_dict[shp])

    n_pred = len(tp_list)
    prec_thred = []
    rec_thred = []
    for i in np.arange(0.0, 1.1, 0.1):
        tp = 0
        idx = [j for j, s in enumerate(score_list) if s > i]
        for it, t in enumerate(tp_list):
            if it in idx and tp_list[it] == 1:
                tp += 1
        prec_thred.append(tp/n_pred)
        rec_thred.append(tp/n_gt)

    ap = compute_ap(np.array(rec_thred), np.array(prec_thred))
    score = 0.6 * ap + 0.4 * iou
    return score


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (np.array).
        precision: The precision curve (np.array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[:-1] != mrec[1:])[0]   #错位比较，前一个元素与其后一个元素比较,np.where()返回下标索引数组组成的元组

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate(valjson_path, train_shiftul_path, train_orimg_dir, val_gtshp_dir, unionshp_dir):
    gt_dict, n_gt = get_gt(valjson_path, train_shiftul_path, train_orimg_dir, val_gtshp_dir)
    print("gt done")
    pred_dict, score_dict = get_pred(unionshp_dir)
    print("pred done")
    tp_dict = cal_tp(gt_dict, pred_dict)
    print("calculate tp done")
    iou = cal_iou(gt_dict, pred_dict, cfg.train_orimg_dir)
    print("calculate iou done")
    score = cal_score(tp_dict, score_dict, iou, n_gt)
    print("calculate score done")
    print_extent = [['score', score]]
    print(tabulate(print_extent, tablefmt='grid'))


if __name__ == '__main__':
    is_dir(cfg.val_gtshp_dir)
    evaluate(cfg.valjson_path, cfg.train_shiftul_path, cfg.train_orimg_dir, cfg.val_gtshp_dir, cfg.unionshp_dir)
    

