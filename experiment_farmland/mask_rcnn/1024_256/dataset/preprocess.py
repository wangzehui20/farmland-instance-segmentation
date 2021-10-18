import os
import shapefile
import numpy as np
import imgaug as ia
from itertools import chain
from skimage import io as skio
from imgaug.augmentables.polys import Polygon
from osgeo import gdal, osr, ogr
from multiprocessing import Pool
from common import bbox_double2int, seg_double2int, get_mean_std, is_lowimg


# 真值以shp的形式提供
def data_process(orimg_path, dstimg_dir, cfg, start=0, shpdir=None):
    # 1 读取栅格数据
    orimg_dataset = gdal.Open(orimg_path)
    orimg_info = TifInfo(orimg_dataset)
    polygon_list = []
    json_list = []
    statis = None

    # 2 影像分块
    clip_list = get_clip_list(orimg_info.width, orimg_info.height, cfg.WIDTH, cfg.HEIGHT, cfg.OVERLAP)

    # 3 对影像对应的实例分割坐标集进行坐标变换
    if shpdir is not None:
        shp_path = os.path.join(shpdir, "{}.shp".format(os.path.basename(orimg_path).split('.')[0]))
        reader = shapefile.Reader(shp_path)

        for sr in reader.shapeRecords():
            geom = sr.shape.__geo_interface__
            feature_points = geom["coordinates"][0]
            xy_points = []
            for lonlat in feature_points:
                xy = lonlat2xy(orimg_dataset, float(lonlat[0]), float(lonlat[1]))
                xy_points.append(xy)
            polygon_list.append(Polygon(xy_points))

    # 4 根据裁剪框对相应的目标框进行处理
    for i, clip in enumerate(clip_list):
        x_min = clip[2]
        y_min = clip[0]
        width = clip[3] - clip[2]
        height = clip[1] - clip[0]

        # 存储裁剪的训练图片
        dstimg_path = os.path.join(dstimg_dir, "{}.png".format(start))

        # 坐标框平移
        polygon_list_shift = list(map(lambda x: x.shift(top=-y_min, left=-x_min), polygon_list))
        psoi = ia.PolygonsOnImage(polygon_list_shift,
                                  shape=(height, width))

        # 剔除及截断坐标框
        psoi_aug = psoi.remove_out_of_image(fully=True, partly=True)
        aug_polygon_list = psoi_aug.polygons

        img_data = orimg_dataset.ReadAsArray(x_min, y_min, width, height).astype(np.uint8)  # 获取分块数据
        img_data = np.transpose(img_data, (1, 2, 0))[:, :, [2, 1, 0]]
        if width != cfg.WIDTH or height != cfg.HEIGHT:
            img_data_pad = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3)).astype(np.uint8)
            img_data_pad[:height, :width, :] = img_data[:height, :width, :]
            img_data = img_data_pad
        if shpdir is not None and is_lowimg(img_data):
            continue
        if statis is None:
            statis = get_mean_std(img_data)
        skio.imsave(dstimg_path, img_data)

        json_dict = {
            "flags": {},
            "shapes": [],
            "imagePath": dstimg_path,
            "imageHeight": height,
            "imageWidth": width
        }
        # 点位数据
        for aug_polygon in aug_polygon_list:
            json_shape = {
                "points": [],
                "shape_type": "polygon",
                "flags": {},
                "area": 0.0,
                "bbox": []
            }
            xx_list = aug_polygon.xx.tolist()
            yy_list = aug_polygon.yy.tolist()
            seg_list = list(chain.from_iterable(zip(xx_list, yy_list)))
            json_shape["segmentation"] = [seg_list]

            x_min = min(xx_list)
            x_max = max(xx_list)
            y_min = min(yy_list)
            y_max = max(yy_list)

            width = x_max - x_min
            height = y_max - y_min

            seg_list = []
            for xx, yy in zip(xx_list, yy_list):
                seg_list.append([xx, yy])
            json_shape["points"] = seg_list
            json_shape["area"] = aug_polygon.area
            json_shape["bbox"] = [x_min, y_min, width, height]
            json_dict["shapes"].append(json_shape)
        json_list.append(json_dict)
        start += 1
    return start, clip_list, json_list, statis


def generate_coco_json(json_lists, cfg):
    json_dict = {"images": [], "annotations": [], "categories": []}
    idx = 0

    # categories_dict
    categories_dict = {}
    categories_dict["supercategory"] = "farm_land"
    categories_dict["id"] = 1
    categories_dict["name"] = "farm_land"
    json_dict["categories"].append(categories_dict)

    for i, json in enumerate(json_lists):
        images_dict = {}

        # images_dict
        images_dict["height"] = cfg.HEIGHT
        images_dict["width"] = cfg.WIDTH
        # 与shift_upper-left顺序一致
        images_dict["id"] = int(os.path.basename(json["imagePath"]).split('.')[0])
        images_dict["file_name"] = json["imagePath"].split('/')[-1]
        json_dict["images"].append(images_dict)

        # annotations_dict
        for shape in json["shapes"]:
            annotations_dict = {}
            annotations_dict["segmentation"] = seg_double2int(shape["segmentation"])
            annotations_dict["iscrowd"] = 0
            annotations_dict["area"] = shape["area"]
            # 与images_dict["id"]顺序相同
            annotations_dict["image_id"] = int(os.path.basename(json["imagePath"]).split('.')[0])
            annotations_dict["bbox"] = bbox_double2int(shape["bbox"])
            annotations_dict["category_id"] = 1
            annotations_dict["id"] = idx
            idx += 1
            json_dict["annotations"].append(annotations_dict)
    return json_dict


def lonlat2xy(dataset, lon, lat):
    '''
    根据地理坐标(经纬度)转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param lon: 经度坐标
    :param lat: 纬度坐标
    :return: 地理坐标(lon,lat)对应的影像图上行列号(row, col)
    '''
    transform = dataset.GetGeoTransform()
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    x_pix = (lon - x_origin) / pixel_width
    y_pix = (lat - y_origin) / pixel_height
    return (x_pix, y_pix)


# 滑动窗口的形式返回裁剪区域
def get_clip_list(width, height, clipw, cliph, overlap):
    start_w = 0
    start_h = 0
    end_w = clipw
    end_h = cliph
    clip_list = []
    while start_h < height:
        if end_h > height:
            end_h = height
        while start_w < width:
            if end_w > width:
                end_w = width
            clip_list.append([start_h, end_h, start_w, end_w])
            if end_w == width: break
            start_w = end_w - overlap
            end_w = start_w + clipw
        if end_h == height: break
        start_h = end_h - overlap
        end_h = start_h + cliph
        start_w = 0
        end_w = clipw
    return clip_list


class GDAL_shp_Data(object):
    def __init__(self, shp_path):
        self.shp_path = shp_path
        self.shp_file_create()

    def shp_file_create(self):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        ogr.RegisterAll()
        driver = ogr.GetDriverByName("ESRI Shapefile")

        # 打开输出文件及图层
        # 输出模板shp，包含待写入的字段信息
        self.outds = driver.CreateDataSource(self.shp_path)
        # 创建空间参考
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        # 创建图层
        self.out_layer = self.outds.CreateLayer("out_polygon", srs, ogr.wkbPolygon)
        field_name = ogr.FieldDefn("scores", ogr.OFTReal)
        self.out_layer.CreateField(field_name)

    def set_shapefile_data(self, polygons, scores):
        for i in range(len(scores)):
            wkt = polygons[i].wkt  # 创建wkt文本点
            temp_geom = ogr.CreateGeometryFromWkt(wkt)
            feature = ogr.Feature(self.out_layer.GetLayerDefn())  # 创建特征
            feature.SetField("scores", scores[i])
            feature.SetGeometry(temp_geom)
            self.out_layer.CreateFeature(feature)
        self.finish_io()

    def finish_io(self):
        del self.outds


def xy2lonlat(dataset, points):
    '''
    多边形图像坐标转经纬度坐标
    '''
    transform = dataset.GetGeoTransform()
    lonlats = np.zeros(points.shape)
    lonlats[:, 0] = transform[0] + points[:, 0] * transform[1] + points[:, 1] * transform[2]
    lonlats[:, 1] = transform[3] + points[:, 0] * transform[4] + points[:, 1] * transform[5]
    return lonlats


class TifInfo():
    def __init__(self, dataset):
        self.dataset = dataset
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.band = self.dataset.RasterCount
        self.type = gdal.GetDataTypeName(self.dataset.GetRasterBand(1).DataType)
        self.transform = self.dataset.GetGeoTransform()
        self.projection = self.dataset.GetProjection()


def data_process_multi(inputs):
    # train
    if len(inputs) == 5:
        zip_inputs = list(zip(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]))
    # val
    elif len(inputs) == 4:
        zip_inputs = list(zip(inputs[0], inputs[1], inputs[2], inputs[3]))
    pool = Pool()
    results = pool.starmap(data_process, zip_inputs)
    pool.close()
    pool.join()
    return results