# encoding: utf-8
import os
import time
from preprocess import generate_coco_json, data_process_multi
from generate_train_val import rename_img
import sys

sys.path.append("..")
from utils.common import is_dir, save_json, get_imglist
from utils.config import Config


def get_inputs(orimg_dir, dstimg_dir, cfg):
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

    inputs = [orimgs_path, dstimgs_dir, cfgs, start_idxs]
    return inputs


def generate_test_data(orimg_dir, dstimg_dir, cfg):
    tif_list = get_imglist(orimg_dir)
    num = 0
    shift_ul = {}
    json_lists = []
    statis_dict = {}

    inputs = get_inputs(orimg_dir, dstimg_dir, cfg)
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


if __name__ == '__main__':
    start_time = time.time()
    cfg = Config()

    is_dir(cfg.test_clpimg_dir)
    is_dir(os.path.dirname(cfg.test_shiftul_path))

    shift_ul, json_lists, statistic_dict, _ = generate_test_data(cfg.test_orimg_dir, cfg.test_clpimg_dir, cfg)
    print("Generate test image successfully")

    rename_img(cfg.test_orimg_dir, cfg.test_clpimg_dir, cfg)
    print("Rename image successfully")

    save_json(cfg.test_statis_path, statistic_dict)
    print("Generate test statistics successfully")

    save_json(cfg.test_shiftul_path, shift_ul)
    print("Generate test shift_ul successfully")

    test_json = generate_coco_json(json_lists, cfg.HEIGHT, cfg.WIDTH)
    save_json(cfg.testjson_path, test_json)
    print("Generate test json successfully")

    end_time = time.time()
    print("time", end_time-start_time)
