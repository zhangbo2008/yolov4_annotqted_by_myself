# coding=utf-8
import os
import sys

sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
import config.yolov4_config as cfg
import cv2
import numpy as np
import random

# from . import data_augment as dataAug
# from . import tools

import utils.data_augment as dataAug
import utils.tools as tools


class Build_Dataset(Dataset): # 这个函数把图片数据转化为pytorch的dataset
    def __init__(self, anno_file_type, img_size=416):
        self.img_size = img_size  # For Multi-training
        if cfg.TRAIN["DATA_TYPE"] == "VOC":
            self.classes = cfg.VOC_DATA["CLASSES"]
        elif cfg.TRAIN["DATA_TYPE"] == "COCO":
            self.classes = cfg.COCO_DATA["CLASSES"]
        else:
            self.classes = cfg.Customer_DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        assert item <= len(self), "index range error"

        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1) #随机抽取幸运观众
        img_mix, bboxes_mix = self.__parse_annotation(
            self.__annotations[item_mix]
        ) # __parse_annotation 里面进行了图片和box 的提升.
        img_mix = img_mix.transpose(2, 0, 1)
# 另外一种数据增强. v4 创新点.
        img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix
#删除中间变量.
        (
            label_sbbox,  # shape:  52,52,3,26 #debug技巧: 知道每一个shape里面每一个分量的物理含义.
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        ) = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return (
            img,
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        )

    def __load_annotations(self, anno_type):

        assert anno_type in [
            "train",
            "test",
        ], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(
            "data", anno_type + "_annotation.txt"
        )
        with open(anno_path, "r") as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)

        return annotations  # 这个就是以后玩的标注数据了.

    def __parse_annotation(self, annotation):
        """
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(" ")

        img_path = anno[0] # 注意cv2里面图片地址不能有中文.
        img_path=img_path.replace("数据集",'')
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, "File Not Found " + img_path
        bboxes = np.array(
            [list(map(float, box.split(","))) for box in anno[1:]]
        )
# 这个函数太牛逼了.直接img,bbox直接全提升了.
        img, bboxes = dataAug.RandomHorizontalFilp()(
            np.copy(img), np.copy(bboxes), img_path
        )
        img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(
            np.copy(img), np.copy(bboxes)
        )

        return img, bboxes

    def __creat_label(self, bboxes): # fastrcnn
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """

        anchors = np.array(cfg.MODEL["ANCHORS"]) # 3个锚点, 下面我们计算anchor的面积> anchors[:,:,0]/anchors[:,:,1]
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides    # strides表示多少个pixel 来表示一个anchor框.
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"] # 一共3个scale的anchor框,
# anchors 的长宽   [[[ 1.25     1.625  ],  [ 2.       3.75   ],  [ 4.125    2.875  ]],, [[ 1.875    3.8125 ],  [ 3.875    2.8125 ],  [ 3.6875   7.4375 ]],, [[ 3.625    2.8125 ],  [ 4.875    6.1875 ],  [11.65625 10.1875 ]]]    下面我们看strides: 是            8 , 16, 32  第一个anchor:    [[ 1.25     1.625  ],  [ 2.       3.75   ],  [ 4.125    2.875  ]]  显然里面的长度都小于8,  第二个anchor     [[ 1.875    3.8125 ],  [ 3.875    2.8125 ],  [ 3.6875   7.4375 ]], 都小于16 ,第三个都小于32.        anchors的长款比例:    anchors[:,:,0]/anchors[:,:,1]  :  [[0.76923077 0.53333333 1.43478261], [0.49180328 1.37777778 0.49579832], [1.28888889 0.78787879 1.14417178]] 是这个9个数值.  看出来都是一个是接近一半的,另外是一个1.5倍, 另外一个0.7左右的. 来涵盖所有情况.
        label = [
            np.zeros(
                (
                    int(train_output_size[i]),
                    int(train_output_size[i]),
                    anchors_per_scale,
                    6 + self.num_classes, # mixup函数. 4个坐标,物品的分类索引,带alpha的置信度
                ) #每一个比例我们玩 多少个预测点,每一个预测点我们玩 3个anchor,每一个anchor我们做一个6+20的分类任务. 6表示长宽高分类号,mix置信度,加 20分类.
            )
            for i in range(3)  # 对比例进行循环. 3代表比例.
        ]
        for i in range(3):
            label[i][..., 5] = 1.0   # mix置信度都是1

        bboxes_xywh = [
            np.zeros((150, 4)) for _ in range(3)
        ]  # Darknet the max_num is 30  # 这个用来存储所有的gt_box 也是对于各个尺度分别算.
        bbox_count = np.zeros((3,)) # 对于3个尺度分别统计.
#===========下面的for就是核心的解析部分.
        for bbox in bboxes: # 对每一个box进行解析
            bbox_coor = bbox[:4]  #坐标
            bbox_class_ind = int(bbox[4])  #分类
            bbox_mix = bbox[5]  #置信度

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,  #中心坐标
                    bbox_coor[2:] - bbox_coor[:2],    #长宽
                ],
                axis=-1,
            )
            # print("bbox_xywh: ", bbox_xywh)
            for j in range(len(bbox_xywh)):# 进行box裁剪
                if int(bbox_xywh[j]) >= self.img_size: #每一个数直, xywh都进行裁剪.
                    differ = bbox_xywh[j] - float(self.img_size) + 1.
                    bbox_xywh[j] -= differ # 裁剪到边.
            bbox_xywh_scaled = (  # 放缩到我们anchor框的级别, 就是除以strides即可.没难度.因为需要每一个维度都除,所以需要得到3*4的, 分别拓展维度即可. 第一个bbox_xywh前面拓展, strides后面拓展.
                1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4)) #bbox_xywh_scaled  3个比例的bbox
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )  # 0.5 for compensation ============这里面我们上面config里面的anchor只是w,h, 对应的xy要在这里面配置. 让他是动态的.也就是真实box的中心坐标.  也就是默认我们坐标预测百分百正确.
                anchors_xywh[:, 2:4] = anchors[i] # anchors 里面存的是宽高.  表示每一个锚点我们计算他的3种宽高比例.
#==============这些东西都是为了计算这个box应该属于哪个anchor来label化的!!!!!!!!!!!!!
                iou_scale = tools.iou_xywh_numpy(  # 然后我们计算 box 和我们 box配anchor的iou吧.  是box 跟3个anchor一起算, 因为anchor 里面每一个配了3个比例,
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3 # 每一个尺度, 有3个anchor ,我们只要里面大于0.3的部分.

                if np.any(iou_mask): # 如果存在物体.因为我们假设3种尺度,起码会存在一个的.绝大部分都会走这个的.
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    ) # 真实的的索引位置.

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox 确实是bug, 表示的是用后面的覆盖了. # 写入最后的label即可. # 这里面是大于0.3的东西我们都进行判断是物体,然后写入进label数据. 根据iou_mask判断.
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh # 把存在的东西写入label里面========注意这里面记录的是非scale的 box坐标!!!!!!!!!!!!!!!也是只写入当大于0.3的时候,否则也都是0.
                    label[i][yind, xind, iou_mask, 4:5] = 1.0  # 写入置信度
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix  #mix置信度
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth #分类置信度

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大 %150循环写入, 最大就是150个物品一个图片. 因为总控开的空间就是150. 内存不够的话,就开小一点.
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh # 注意这里面依然使用的是绝对坐标. 不进行scale化.
                    bbox_count[i] += 1  # 做统计的.第几个anchor里面有多少个box

                    exist_positive = True

            if not exist_positive: #基本不走下面流程. 表示当前box 不匹配任何一个anchor
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)# 首先我们找到iou最大的.
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)  # 找到detect 和 锚点.
#然后下面就都一样了. 区别是只写入一个best_anchor的索引.
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label # lable是一个数组,里面有3个, 3=3的赋值.
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == "__main__":

    voc_dataset = Build_Dataset(anno_file_type="train", img_size=448)
    dataloader = DataLoader(
        voc_dataset, shuffle=True, batch_size=1, num_workers=0
    )

    for i, (
        img,
        label_sbbox,
        label_mbbox,
        label_lbbox,
        sbboxes,
        mbboxes,
        lbboxes,
    ) in enumerate(dataloader):
        if i == 0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)

            if img.shape[0] == 1:
                labels = np.concatenate(
                    [
                        label_sbbox.reshape(-1, 26),
                        label_mbbox.reshape(-1, 26),
                        label_lbbox.reshape(-1, 26),
                    ],
                    axis=0,
                )
                labels_mask = labels[..., 4] > 0
                labels = np.concatenate(
                    [
                        labels[labels_mask][..., :4],
                        np.argmax(
                            labels[labels_mask][..., 6:], axis=-1
                        ).reshape(-1, 1),
                    ],
                    axis=-1,
                )

                print(labels.shape)
                tools.plot_box(labels, img, id=1)
