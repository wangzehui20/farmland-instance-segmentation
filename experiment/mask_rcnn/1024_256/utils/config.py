class Config():
    def __init__(self):
        self.HEIGHT = 1024
        self.WIDTH = 1024
        self.OVERLAP = 256
        self.SEED = 40
        self.RATIO = 0.8
        self.IOU_THRED = 0.6
        self.RMBACKGROUND_THRED = 0.2   # background threshold  
        self.MERGE_THRED = 0.7
        self.SCORE_THRED = 0
        # self.RMOVERLAP_THRED = 0.9
        # self.REGUL_THRED = 0.8
        self.IMAGE_STEP = 10000   # multi process
        
        self.FILE_NAME = "1024_256"
        self.EXP_NAME = "mask_rcnn"
        self.EPOCH = "epoch_11"
        self.WORK_DIRS = "work_dirs"
        self.MODE = "test-dev"   # val or test-dev
        
        self.ORI_DIR = "/data/data/farm_land/origin"
        self.COCO_DIR = "/data/data/farm_land/clip"
        self.RES_DIR = "/data/data/farm_land/result"

        self.COCO_BASEDIR = rf"{self.COCO_DIR}/{self.EXP_NAME}/{self.FILE_NAME}"
        self.RES_BASEDIR = rf"{self.RES_DIR}/{self.EXP_NAME}/{self.FILE_NAME}/{self.WORK_DIRS}"
        
        # train and val
        self.train_orimg_dir = rf"{self.ORI_DIR}/train/image"
        self.train_orilabel_dir = rf"{self.ORI_DIR}/train/label"

        self.annotations_dir = rf"{self.COCO_BASEDIR}/annotations"
        self.train_clpimg_dir = rf"{self.COCO_BASEDIR}/train"
        self.val_clpimg_dir = rf"{self.COCO_BASEDIR}/val"

        self.trainjson_path = rf"{self.COCO_BASEDIR}/annotations/train.json"
        self.valjson_path = rf"{self.COCO_BASEDIR}/annotations/val.json"
        self.train_shiftul_path = rf"{self.COCO_BASEDIR}annotations/train_shift_ul.json"
        self.train_statis_path = rf"{self.COCO_BASEDIR}/annotations/train_statistics.json"                                          
        
        # test
        self.test_orimg_dir = rf"{self.ORI_DIR}/test/image"
        self.test_clpimg_dir = rf"{self.COCO_BASEDIR}/test"
        self.testjson_path = rf"{self.COCO_BASEDIR}/annotations/test.json"
        self.test_shiftul_path = rf"{self.COCO_BASEDIR}/annotations/test_shift_ul.json"  # record upper-left location
        self.test_statis_path = rf"{self.COCO_BASEDIR}/annotations/test_statistics.json"
        
        # result
        self.pred_path = rf"{self.RES_BASEDIR}/output/results/seg/mask_rcnn_{self.MODE}_results_{self.EPOCH}.segm.json"

        self.outshp_dir = rf"{self.RES_BASEDIR}/output/out_shp/{self.MODE}/{self.EPOCH}_thred_0"
        self.unionshp_dir = rf"{self.RES_BASEDIR}/output/union_shp_iou/{self.MODE}/{self.EPOCH}"
        
        # demo
        self.val_demo_dir = rf"{self.RES_BASEDIR}/output/val_demo"
        self.val_gt_dir = rf"{self.RES_BASEDIR}/output/val_gt"
        self.val_gtshp_dir = rf"{self.RES_BASEDIR}/output/val_gtshp/{self.EPOCH}"
