# encoding: utf-8
import os
import random
import shutil
from preprocess import generate_coco_json, data_process_multi
import time
from common import is_dir, save_json, get_imglist
from config import Config


def move_val_img(trainimg_dir, valimg_dir, val_list):
    img_list = os.listdir(trainimg_dir)
    for img in img_list:
        if int(img.split('.')[0]) in val_list:
            oripath = os.path.join(trainimg_dir, img)
            dstpath = os.path.join(valimg_dir, img)
            shutil.move(oripath, dstpath)


# ------------------------------------------------
# single process
# ------------------------------------------------

def split_train_val(num):
    seed = 0
    ratio = 0.7

    num_list = [n for n in range(num)]
    random.seed(seed)
    random.shuffle(num_list)
    split_idx = int(len(num_list) * ratio)
    train_list = num_list[:split_idx]
    val_list = num_list[split_idx:]
    return train_list, val_list


def get_train_val_json(val_list, json_lists):
    train_json = []
    val_json = []
    for i, json in enumerate(json_lists):
        if i in val_list:
            val_json.append(json)
        else:
            train_json.append(json)
    return train_json, val_json


# def get_train_data(orimg_dir, dstimg_dir, orilabel_dir, cfg):
#     tif_list = get_imglist(orimg_dir)
#     start_idx = 0
#     shift_ul = {}
#     json_lists = []
#     statis_dict = {}
#     # first put all clip image on train file then move random image into val file
#     for tif in tqdm(tif_list, total=len(tif_list)):
#         tif_path = os.path.join(orimg_dir, tif)
#         end_idx, clip_list, json_list, statis = data_process(tif_path, dstimg_dir, cfg,
#                                                                       start_idx, orilabel_dir)
#         json_lists.extend(json_list)
#         statis_dict[tif] = statis
#         for j in range(len(json_list)):
#             shift_ul["{}.png".format(start_idx + j)] = (tif, clip_list[j][2], clip_list[j][0])   # (tif name, upper-left x, upper-left y)
#         start_idx = end_idx
#     return shift_ul, json_lists, statis_dict, start_idx


# ------------------------------------------------
# multi process
# ------------------------------------------------

def get_inputs(orimg_dir, dstimg_dir, cfg, orilabel_dir):
    tif_list = get_imglist(orimg_dir)

    orimgs_path = [os.path.join(orimg_dir, tif) for tif in tif_list]
    # tmp dst_imgs_dir
    dstimgs_dir = []
    for i in range(len(tif_list)):
        tmpath = "{}_{}".format(dstimg_dir, i)
        is_dir(tmpath)
        dstimgs_dir.append(tmpath)
    cfgs = [cfg for i in range(len(tif_list))]
    start_idxs = [i * cfg.IMAGE_STEP for i in range(len(tif_list))]
    orilabels_dir = [orilabel_dir for i in range(len(tif_list))]

    inputs = [orimgs_path, dstimgs_dir, cfgs, start_idxs, orilabels_dir]
    return inputs


def get_train_data(orimg_dir, dstimg_dir, orilabel_dir, cfg):
    # start_idx = 0
    num = 0
    json_lists = []
    statis_dict = {}
    tif_list = get_imglist(orimg_dir)
    shift_ul = {}

    inputs = get_inputs(orimg_dir, dstimg_dir, cfg, orilabel_dir)
    results = data_process_multi(inputs)
    for i in range(len(results)):
        # results[i]: start_idx, clip_box_list, json_list, statistic
        tif = tif_list[i]
        for j in range(len(results[i][2])):
            clip_list = results[i][1]
            img_name = "{}.png".format(num)   # rename
            shift_ul[img_name] = (tif, clip_list[j][2], clip_list[j][0])   # (tif name, upper-left x, upper-left y)
            results[i][2][j]["imagePath"] = os.path.join(dstimg_dir, img_name)
            num += 1
        statis_dict[tif] = results[i][3]
        json_lists.extend(results[i][2])
    return shift_ul, json_lists, statis_dict, num


def rename_img(orimg_dir, dstimg_dir, cfg):
    tif_list = get_imglist(orimg_dir)
    dstimgs_dir = ["{}_{}".format(dstimg_dir, i) for i in range(len(tif_list))]
    tmpnum = 0
    for i in range(len(tif_list)):
        if i == 0:
            os.rename(dstimgs_dir[0], dstimg_dir)
            tmpnum += len(os.listdir(dstimg_dir))
        else:
            for j in range(len(os.listdir(dstimgs_dir[i]))):
                oripath = os.path.join(dstimgs_dir[i], "{}.png".format(i*cfg.IMAGE_STEP+j))
                dstpath = os.path.join(dstimg_dir, "{}.png".format(tmpnum))
                shutil.move(oripath, dstpath)
                tmpnum += 1
            os.rmdir(dstimgs_dir[i])


if __name__ == '__main__':
    start_time = time.time()
    cfg = Config()

    annotations_dir = rf"{cfg.COCO_BASEDIR}/annotations"
    train_orimg_dir = rf"{cfg.ORI_DIR}/train/image"
    train_clpimg_dir = rf"{cfg.COCO_BASEDIR}/train"
    train_orilabel_dir = rf"{cfg.ORI_DIR}/train/label"
    train_statis_path = rf"{cfg.COCO_BASEDIR}/annotations/train_statistics.json" 
    train_shiftul_path = rf"{cfg.COCO_BASEDIR}/annotations/train_shiftul.json"
    trainjson_path = rf"{cfg.COCO_BASEDIR}/annotations/train.json"
    val_clpimg_dir = rf"{cfg.COCO_BASEDIR}/val"
    valjson_path = rf"{cfg.COCO_BASEDIR}/annotations/val.json"
    pred_path = rf"{cfg.RES_BASEDIR}/output/results/seg/mask_rcnn_{cfg.MODE}_results_{cfg.EPOCH}.segm.json"
    is_dir(annotations_dir)
    is_dir(train_clpimg_dir)
    is_dir(val_clpimg_dir)
    is_dir(os.path.dirname(pred_path))

    shift_ul, json_lists, statis_dict, start_idx = get_train_data(train_orimg_dir, train_clpimg_dir, train_orilabel_dir, cfg)
    print("Generate train image and label successfully")

    save_json(train_statis_path, statis_dict)
    print("Generate train statistics successfully")

    save_json(train_shiftul_path, shift_ul)
    print("Generate train shift_ul successfully")

    # val json
    train_list, val_list = split_train_val(start_idx)
    train_json, val_json = get_train_val_json(val_list, json_lists)

    rename_img(train_orimg_dir, train_clpimg_dir, cfg)
    print("Rename image successfully")

    # val image
    move_val_img(train_clpimg_dir, val_clpimg_dir, val_list)
    print("Generate val image and label successfully")

    train_json = generate_coco_json(train_json, cfg)
    val_json = generate_coco_json(val_json, cfg)

    save_json(trainjson_path, train_json)
    print("Generate train json successfully")
    save_json(valjson_path, val_json)
    print("Generate val json successfully")

    end_time = time.time()
    print("time", end_time-start_time)
