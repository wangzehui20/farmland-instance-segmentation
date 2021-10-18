class Config():
    def __init__(self):
        self.HEIGHT = 1024
        self.WIDTH = 1024
        self.OVERLAP = 256
        self.SEED = 40
        self.RATIO = 0.8
        self.IOU_THRED = 0.6
        self.MERGE_THRED = 0.7
        self.SCORE_THRED = 0
        # self.RMOVERLAP_THRED = 0.9
        self.IMAGE_STEP = 10000   # multi process
        
        self.FILE_NAME = "1024_256"
        self.EXP_NAME = "mask_rcnn"
        self.EPOCH = "best"   # epoch11
        self.WORK_DIRS = "work_dirs"
        self.MODE = "test-dev"   # val or test-dev
        
        self.ORI_DIR = "/data/data/farm_land/origin"
        self.COCO_DIR = "/data/data/farm_land/clip"
        self.RES_DIR = "/data/data/farm_land/result"

        self.COCO_BASEDIR = rf"{self.COCO_DIR}/{self.EXP_NAME}/{self.FILE_NAME}"
        self.RES_BASEDIR = rf"{self.RES_DIR}/{self.EXP_NAME}/{self.FILE_NAME}/{self.WORK_DIRS}"
