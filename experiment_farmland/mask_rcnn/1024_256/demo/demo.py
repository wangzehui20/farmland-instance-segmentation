from mmdet.models.detectors.base import BaseDetector
from mmdet.apis import init_detector, inference_detector
import os
import cv2
import torch
import numpy as np 
from tqdm import tqdm
from demo_prepare import show_det_result
from demo_prepare_gt import show_gt_result
from ..dataset.common import is_dir, open_json
from ..dataset.config import Config
# BaseDetector.show_result


def demo(img_dir, demo_dir, val_demonopost_dir, valjson_path, val_gt_dir, model, cfg):
    # predict
    img_list = os.listdir(img_dir)
    imgs_path = []
    for img in img_list:
        imgs_path.append(os.path.join(img_dir, img))
    for img_path in tqdm(imgs_path, total=len(imgs_path)):
        # gt
        anns, val_info = get_val_info(valjson_path)
        if os.path.basename(img_path) not in val_info.keys(): continue
        bbox, segm = get_gt(anns, val_info, os.path.basename(img_path), cfg)
        gt = ([bbox],[segm])
        show_gt_result(img=img_path, result=gt, show=True, score_thr=cfg.SCORE_THRED,
                                    out_file=os.path.join(val_gt_dir, os.path.basename(img_path)))

        # pred
        result = inference_detector(model, img_path)
        ## no post process
        show_det_result(img=img_path, result=result, show=True, score_thr=cfg.SCORE_THRED,
                                    out_file=os.path.join(val_demonopost_dir, os.path.basename(img_path)))
        ## post process
        result = pred_post(result)
        show_det_result(img=img_path, result=result, show=True, score_thr=cfg.SCORE_THRED,
                                    out_file=os.path.join(demo_dir, os.path.basename(img_path)))


def get_val_info(path):
    val_json = open_json(path)
    anns = val_json["annotations"]
    val_info = {}
    for i, ann in enumerate(anns):
        img_name = "{}.png".format(ann["image_id"])
        if img_name not in val_info.keys(): val_info[img_name] = []
        val_info[img_name].append(i)
    return anns, val_info


def get_gt(annotations, info, name, cfg):
    bbox= []
    segm = []
    for i in info[name]:
        annotations[i]["bbox"].append(1)
        bbox.append(annotations[i]["bbox"])
        img = get_gt_img(annotations[i]["segmentation"][0], cfg)
        segm.append(img)
    return np.array(bbox, dtype=np.float32), segm


def get_gt_img(seg, cfg):
    img = np.zeros((cfg.HEIGHT, cfg.WIDTH))
    seg_array = np.array([[seg[i:i+2][0], seg[i:i+2][1]] for i in range(0, len(seg), 2)])
    cv2.fillPoly(img, [seg_array], 1)
    img = np.equal(img, 1)
    return img


# result of post process
def pred_post(result):
    score = []
    discard = []
    for i in range(len(result[0][0])):
        score.append(result[0][0][i][-1])
    rsort_idx = np.argsort(score)[::-1]

    for i in range(len(rsort_idx)-1):
        for j in range(i+1, len(rsort_idx)):
            polygon = result[1][0][rsort_idx[i]].astype(np.uint8)
            polygon_pending = result[1][0][rsort_idx[j]].astype(np.uint8)
            inters = np.sum(polygon * polygon_pending)
            union = np.sum((polygon==1)*(polygon_pending==0)) + np.sum((polygon==0)*(polygon_pending==1)) + inters
            if inters / union > 0.5 or inters / np.sum(polygon_pending) > cfg.MERGE_THRED:
                discard.append(rsort_idx[j])
                result[1][0][rsort_idx[i]] = ((polygon==1)*(polygon_pending==0)+(polygon==0)*(polygon_pending==1)+(polygon==1)*(polygon_pending==1)).astype(np.bool)
    bbox = []
    segm = []
    for i in range(len(result[0][0])):
        if i not in discard:
            bbox.append(result[0][0][i])
            segm.append(result[1][0][i])
    return ([np.array(bbox)], [segm])


if __name__ == '__main__':
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_clpimg_dir = rf"{cfg.COCO_BASEDIR}/val"
    val_demonopost_dir = rf'{cfg.RES_BASEDIR}/output/val_demo_nopost'
    val_demo_dir = rf"{cfg.RES_BASEDIR}/output/val_demo"
    valjson_path = rf"{cfg.COCO_BASEDIR}/annotations/val.json"
    val_gt_dir = rf"{cfg.RES_BASEDIR}/output/val_gt"
    is_dir(val_demonopost_dir)
    is_dir(val_demo_dir)

    config_file = r'../code/mask_rcnn_res50/mask_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = r'/data/data/farm_land/best_model/best.pth'

    model = init_detector(config_file, checkpoint_file, device)
    demo(val_clpimg_dir, val_demo_dir, val_demonopost_dir, valjson_path, val_gt_dir, model, cfg)
