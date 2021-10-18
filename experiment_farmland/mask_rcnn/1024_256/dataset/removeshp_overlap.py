import os
from numpy.core.fromnumeric import argsort
import shapefile
import numpy as np
import time
from shapely.geometry import Polygon
from preprocess import GDAL_shp_Data, lonlat2xy
from tqdm import tqdm
from osgeo import gdal
from shapely.ops import unary_union
from multiprocessing import Pool
from common import is_dir, get_shplist
from config import Config

os.environ['PROJ_LIB'] = '/opt/conda/pkgs/proj-6.2.1-haa6030c_0/share/proj'


def is_union(polygon_list):
    return True if unary_union(polygon_list).geom_type == 'Polygon' else False


# ------------------------------------------------
# polygons union directly
# ------------------------------------------------

# def polygon_union_direct(shp_path):
#     reader = shapefile.Reader(shp_path)
#     polygon_list = []
#     for sr in tqdm(reader.shapeRecords(), total=len(reader.shapeRecords())):
#         geom = sr.shape.__geo_interface__
#         feature_points = geom["coordinates"][0]
#         lonlat_list = []
#         for lonlat in feature_points:
#             lonlat_list.append((float(lonlat[0]), float(lonlat[1])))
#         polygon_list.append(Polygon(lonlat_list))
#     union = unary_union(polygon_list)
#     return union


# ------------------------------------------------
# bbox nms
# ------------------------------------------------

# def cal_iou(bbox, scores, cfg):
#     x1 = bbox[:, 0]
#     y1 = bbox[:, 1]
#     x2 = bbox[:, 2]
#     y2 = bbox[:, 3]
#     areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort()[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.maximum(x2[i], x2[order[1:]])
#         yy2 = np.maximum(y2[i], y2[order[1:]])
#         inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
#         iou = inter / (areas[i] + areas[order[1:]] - inter)
#         inds = np.where(iou <= cfg.IOU_THRED)[0]
#         order = order[inds + 1]
#     return keep


# def bbox_iou(test_orimg_dir, shp_path, cfg):
#     orimg_path = os.path.join(test_orimg_dir, "{}.tif".format(os.path.basename(shp_path).split('.')[0]))
#     dataset = gdal.Open(orimg_path)
#     reader = shapefile.Reader(shp_path).shapeRecords()
#     # order scores
#     # reader.sort(key=lambda rd: rd.record["scores"])
#     polygon_list = []
#     scores_list = []
#     bbox_list = []
#     lonlat_list = []
#     for i in range(len(reader)):
#         # lonlat to imagexy
#         # 1.points
#         points = reader[i].shape.points
#         imagexy_point = []
#         for point in points:
#             imagexy_point.append(lonlat2xy(dataset, point[0], point[1]))
#         polygon_list.append(imagexy_point)
#         # 2.bbox
#         bbox = reader[i].shape.bbox
#         point_tl = lonlat2xy(dataset, bbox[0], bbox[1])
#         point_br = lonlat2xy(dataset, bbox[2], bbox[3])
#         bbox_list.append([point_tl[0], point_tl[1], point_br[0], point_br[1]])
#         # 3.scores
#         scores_list.append(reader[i].record[0])
#     bbox = np.array(bbox_list, dtype=np.float64)
#     scores = np.array(scores_list, dtype=np.float64)
#     polygon = np.array(polygon_list)
#     keep = cal_iou(bbox, scores, cfg)
#     polygon_list = polygon[keep].tolist()
#     scores_list = scores[keep].tolist()
#     for polygon in polygon_list:
#         lonlat_list.append(Polygon(polygon))
#     return lonlat_list, scores_list


# ------------------------------------------------
# merge and remove polygon
# ------------------------------------------------

# 1. merge overlap polygons and remove most included polygon
def cal_union_update(polygon_list, cfg):
    # update union polygon
    discard_idx = []
    # reversed oreder
    for i in tqdm(reversed(range(1, len(polygon_list))), total=len(polygon_list) - 1):
        if i in discard_idx: continue
        # 先和score低的polygon判断是否重叠
        for j in reversed(range(0, i - 1)):
            if is_union([polygon_list[i], polygon_list[j]]) and rm_overlap_iou(polygon_list[i], polygon_list[j], cfg.IOU_THRED):
                polygon_list[i] = unary_union([polygon_list[i], polygon_list[j]])
                discard_idx.append(j)

        # # 再和之前处理的polygon判断是否重叠
        # keep_idx = []
        # # 把score最高的放到最后
        # for k in range(i + 1, len(polygon_list)):
        #     if k not in discard_idx and is_union([polygon_list[i], polygon_list[k]]) and rm_overlap_iou(polygon_list[i], polygon_list[k], cfg.IOU_THRED):
        #         keep_idx.append(k)
        #         polygon_list[i] = unary_union([polygon_list[i], polygon_list[k]])
        # if len(keep_idx) == 1:
        #     polygon_list[keep_idx[0]] = polygon_list[i]
        #     discard_idx.append(i)
        # elif len(keep_idx) > 1:
        #     for ki in range(len(keep_idx) - 1):
        #         discard_idx.append(keep_idx[ki])
        #     polygon_list[keep_idx[len(keep_idx) - 1]] = polygon_list[i]
        #     discard_idx.append(i)

        # 使用递归合并，和上面得到的结果一样
        # n_discard = len(discard_idx)
        # polygon_list, discard_idx = cal_union_previous(polygon_list, keep_idx, discard_idx, i)
        # n_discard_new = len(discard_idx)
        # if n_discard_new - n_discard == 0: 
        #     keep_idx.append(i)
        # elif n_discard_new - n_discard > 1:
        #     for k in range(n_discard+1, n_discard_new):
        #         keep_idx.remove(discard_idx[k]) 
    return discard_idx, polygon_list


# # 递归合并之前处理过的polygon
# def cal_union_previous(polygon_list, keep_idx, discard_idx, cur):
#     for k in keep_idx:
#         if k > cur and is_union([polygon_list[cur], polygon_list[k]]):
#             polygon_list[k] = unary_union([polygon_list[cur], polygon_list[k]])
#             discard_idx.append(cur)
#             polygon_list, discard_idx = cal_union_previous(polygon_list, keep_idx, discard_idx, k)
#             break
#     return polygon_list, discard_idx


# 2. remove lower score overlap polygons
def rm_union_update(polygon_list, score_list):
    discard_idx = []
    update_polygon_list = []
    update_score_list= []
    for i in tqdm(reversed(range(len(polygon_list))), total=len(polygon_list)):
        for j in reversed(range(0, i - 1)):
            if is_union([polygon_list[i], polygon_list[j]]) and rm_overlap(polygon_list[i], polygon_list[j], cfg.RMOVERLAP_THRED):
                discard_idx.append(j)
    for i in range(len(polygon_list)):
        if i not in discard_idx:
            update_polygon_list.append(polygon_list[i])
            update_score_list.append(score_list[i])
    return update_polygon_list, update_score_list


def rm_overlap_iou(polygon, polygon_pending, thred=0.5):
    # prevent self-intersection
    polygon = polygon.buffer(1e-10)
    polygon_pending = polygon_pending.buffer(1e-10)

    inter = polygon.intersection(polygon_pending)
    area_inter = inter.area
    area_union = polygon.area + polygon_pending.area - area_inter

    # high iou and intersection mostly include in polygon_pending 
    return True if area_inter / area_union > thred or area_inter / polygon_pending.area > cfg.MERGE_THRED else False
        

def rm_overlap(polygon, polygon_pending, thred=0.8):
    polygon = polygon.buffer(1e-10)
    polygon_pending = polygon_pending.buffer(1e-10)
    inter = polygon.intersection(polygon_pending)
    return True if inter.area / polygon.area > thred else False


def polygon_union(shp_path, cfg):
    reader = shapefile.Reader(shp_path).shapeRecords()
    reader.sort(key=lambda rd: rd.record["scores"])
    polygon_list = []
    union_polygon_list = []
    score_list = []
    for i in range(len(reader)):
        points = reader[i].shape.points
        polygon_list.append(Polygon(points))

    # only keep highest score
    # discard_idx = cal_union(polygon_list)

    print("start merge overlap polygons and remove most included polygon...")
    discard_idx, polygon_list = cal_union_update(polygon_list, cfg)
    for i in range(len(reader)):
        if i not in discard_idx:
            union_polygon_list.append(polygon_list[i])
            score_list.append(reader[i].record[0])

    # print("start remove lower score overlap polygon...")
    # union_polygon_list, score_list = rm_union_update(union_polygon_list, score_list)

    return union_polygon_list, score_list


# ------------------------------------------------
# 处理重叠只保留最高的score
# ------------------------------------------------

# def cal_union(polygon_list):
#     discard_idx = []
#     # reversed oreder
#     for i in tqdm(reversed(range(1, len(polygon_list))), total=len(polygon_list) - 1):
#         if i in discard_idx: continue
#         for j in reversed(range(0, i - 1)):
#             if is_union([polygon_list[i], polygon_list[j]]): discard_idx.append(j)
#     return discard_idx


def remove_overlap(outshp_dir, unionshp_dir):
    shp_list = get_shplist(outshp_dir)
    for i, shp in tqdm(enumerate(shp_list), total=len(shp_list)):
        shp_path = os.path.join(outshp_dir, shp)
        outshp_path = os.path.join(unionshp_dir, shp)
        # polygon_list,  scores_list = bbox_iou(test_orimg_dir, shp_path, cfg)
        polygon_list, scores_list = polygon_union(shp_path, cfg)
        shp_data = GDAL_shp_Data(outshp_path)
        shp_data.set_shapefile_data(polygon_list, scores_list)


def remove_overlap_multi(outshp_dir, unionshp_dir, cfg):
    shp_list = get_shplist(outshp_dir)
    shp_paths = [[os.path.join(outshp_dir, shp)] for shp in shp_list]
    cfgs = [cfg for i in range(len(shp_list))]
    pool = Pool()
    results = pool.starmap(polygon_union, list(zip(shp_paths, cfgs)))
    pool.close()
    pool.join()
    for i, shp in enumerate(shp_list):
        outshp_path = os.path.join(unionshp_dir, shp)
        shp_data = GDAL_shp_Data(outshp_path)
        shp_data.set_shapefile_data(results[i][0], results[i][1])


if __name__ == '__main__':
    start_time = time.time()
    cfg = Config()

    unionshp_dir = rf"{cfg.RES_BASEDIR}/output/union_shp_iou/{cfg.MODE}/{cfg.EPOCH}"
    outshp_dir = rf"{cfg.RES_BASEDIR}/output/out_shp/{cfg.MODE}/{cfg.EPOCH}"
    is_dir(unionshp_dir)
    # remove_overlap(outshp_dir, union_shp_dir)
    remove_overlap_multi(outshp_dir, unionshp_dir, cfg)

    end_time = time.time()
    print("time", end_time-start_time)
