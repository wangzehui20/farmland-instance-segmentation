#!/usr/bin/env bash


# 1x res50
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29401 \
train.py 1024_256/code/mask_rcnn_res50/mask_rcnn_r50_fpn_1x_coco.py --launcher pytorch \
--work-dir /data/data/farm_land/result/mask_rcnn/1024_256/work_dirs/checkpoints