# coding=utf-8
import cv2
import random
import numpy as np


class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes, img_path):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            # img = np.fliplr(img)
            img = img[:, ::-1, :] #  [xmin, ymin, xmax, ymax, str(class_id)] #这个地方要记住物理含义!!!!!!!!!!!!!!!! 所以 第一位读是batch_size, 第二维度是这5个数, 第一个维度:不改变,第二个维度的0,2索引位置互换.也就是xmin和xmax互换即可. 因为你图片水平翻转后用图片宽度剪一下即可显然就是这么算的. 可以自己画图笔画一下就懂了.
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        return img, bboxes


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
#首先计算图片最大的框.
            max_bbox = np.concatenate(
                [  #因为我们每一个图片里面有很多个物体, #对于所有物体,也就是axis=0,我们算min和max
                    np.min(bboxes[:, 0:2], axis=0),  #所有xmin的最小值. 所有x
                    np.max(bboxes[:, 2:4], axis=0),  #xmax和ymax的最大值.
                ],
                axis=-1,
            )#所以最后得到的这个max_bbox就是图片中涵盖所有框的一个大框. 他里面有4个数值. 还是表示xmin,ymin,xmax, ymax
            max_l_trans = max_bbox[0]       # 表示xmin到左边框的距离
            max_u_trans = max_bbox[1]       # 向上的距离
            max_r_trans = w_img - max_bbox[2] #向右的距离
            max_d_trans = h_img - max_bbox[3] #向下的距离.

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            ) # 0 到 max_l_trans 从这个代码说明了crop之后的图片必须要包含之前图片中的所有物体. 也就是物品不能被切割. 这也是符合我们逻辑的点. 得到的数值就是处理后图片的大小索引.
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w_img, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h_img, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )# 这3个同理.保证被检测物品不被切割.
#截取就是切片就好了.
            img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
#对于box的crop运算.我们直接减少切割的最小坐标即可. 自己画画就懂了.
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes


class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )#下面玩仿射变换. 跟上面crop一样代码
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]
#然后玩一个直线. 仿射就是平移加旋转变换.我们先算平移. 就是tx,ty.  random.uniform 包含最低的不包含最高的.跟list切片一个意思. 注意这里面都留了一个像素.这个是必须的. 解释如下. 假设我们 max_l_trans ==1 , 那么我们的maxbox 的线就画在了1像素上.这时候我们可以切割掉的部分是0像素.因为我们要留一个边.不能上来就是黑线.
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]]) # M变幻矩阵是2*3的. 这里面1001组成的矩阵说明是id,也就是旋转角度为0.不进行旋转, tx,ty就是平移呗.
            img = cv2.warpAffine(img, M, (w_img, h_img))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx  # bbox也进行平移即可.
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty # 我估计仿射里面如果放入旋转,bbox的计算太复杂了.所以他没写.
        return img, bboxes


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org, w_org, _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(
            1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org
        )
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh : resize_h + dh, dw : resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class Mixup(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if random.random() > self.p: # 这是另外一种数据提升方法, 就是图片组合.
            lam = np.random.beta(1.5, 1.5) #  img_org 是一个图片 img_mix 是另一个图片
            img = lam * img_org + (1 - lam) * img_mix # 图片重影.........
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1
            )
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1
            )
            bboxes = np.concatenate([bboxes_org, bboxes_mix])

        else:
            img = img_org
            bboxes = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1
            )

        return img, bboxes # 最后一个位置的bboxes是补的置信度.


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes
