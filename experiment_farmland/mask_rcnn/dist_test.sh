#!/usr/bin/env bash


mkdir -vp /data/data/farm_land/result/mask_rcnn/1024_256/work_dirs/output/results/seg

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
test.py 1024_256/code/mask_rcnn_res50/mask_rcnn_r50_fpn_1x_coco.py /data/data/farm_land/best_model/epoch_11.pth \
--launcher pytorch \
--format-only --options "jsonfile_prefix=/data/data/farm_land/result/mask_rcnn/1024_256/work_dirs/output/results/seg/mask_rcnn_test-dev_results_epoch11"