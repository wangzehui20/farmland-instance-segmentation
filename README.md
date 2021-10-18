# Farmland Segmentation on IFLYTEK challenge

In this [Farmland Segmentation Challenge](http://challenge.xfyun.cn/topic/info?type=plot-extraction-2021), I only use **Mask RCNN + FPN** on [MMDetection](https://github.com/open-mmlab/mmdetection) to achieve **rank 3**

My code is mainly located in `experiment_farmland` folder

### Introduction

This competition aims to extract farmland segmentation from large remote-sensing images, so i will introduce my plan explicitly later

MMDetection is the most convenient and useful open source framework to learn deep learning, you could achieve better scores easily from most projects and study its poetical code. **Appreciating to the contributors of MMDetection!**

 ### Installation

Firstly, you need to configure MMDetection environment

* I recommend you to follow [official guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
* and also refer to requirements from `experiment_farmland/requirements.txt`
*  `conda install gdal`

### Custom dataset

Competition official gives **large remote-sensing images that needs to be clipped into the small images** and MMDetection requires **COCO dataset format** and so on

I implement above content by sliding window overlapping clipping in `experiment_farmland\mask_rcnn\1024_256\dataset`, image size is 1024 and overlap is 256 pixels

Your dataset transformed to coco dataset after that **soft linked** `experiment_farmland\mask_rcnn\data_1024_256`

Then, replace classes with [your classes](https://zhuanlan.zhihu.com/p/101983661) before training:

* `mmdet/datasets/coco.py`

  ```
  CoCoDataset(CustomDataset): CLASSES = ('farm_land')
  ```

* `mmdet/core/evaluation/class_names.py`

  ```
  def coco_classes(): return ['farm_land']
  ```

* `experiment_farmland\mask_rcnn\1024_256\code\mask_rcnn_res50\coco_instance.py`

  ```
  classes=('farm_land')
  ```

### Train and Test

`experiment_farmland\mask_rcnn`

```
# Train
bash dist_train.sh
# Test
bash dist_test.sh
```

### Tricks

* GIoU Loss
* Soft NMS

### Post process

**Q**: Segmentation exist overlapping which may generate from **inferior inference results and resume origin remote-sensing image from small clipped images**. 

**A**: My solution is that union **IoU > 0.5** or **intersection/polygon(low score) > 0.7** which help me improve score approximately **5 points** in semi-finals

### Demo

``experiment_farmland\mask_rcnn\1024_256\demo\demo.py`

It could generate **gt**,  **pred**,  **pred after post process** images to analysis problem

* **pred and gt**, better in regular farmland

![avatar](https://github.com/wangzehui20/farmland-instance-segmentation/blob/master/experiment_farmland/mask_rcnn/1024_256/demo/pred_gt.png?raw=true)

* **pred and predpost**

![avatar](https://github.com/wangzehui20/farmland-instance-segmentation/blob/master/experiment_farmland/mask_rcnn/1024_256/demo/pred_predpost.png?raw=true)

Also, above demo contrast tools refers to my another work, [BatchLabelCrop](https://github.com/wangzehui20/BatchLabelCrop)

**:smile:**:stuck_out_tongue_winking_eye: :kissing_smiling_eyes:

