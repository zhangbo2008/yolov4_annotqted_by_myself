import sys

sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
import config.yolov4_config as cfg


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(
            torch.abs(target - torch.sigmoid(input)), self.__gamma
        ) # 等于在交叉熵的基础上, 做一个平滑处理而已.  乘以的系数就是 差**2     所以我们理解就是差别比较大的时候,乘以了 差的平方,让loss变小了. 差别比较小的时候,也让loss变小.所以整个曲线就向上鼓起来了. 让loss整体变小. 那么比例如何呢. 我们计算一下   假设loss=10, 所以我们的差比如0.9 也就是说 真实样本比如1,我们只预测了他概率0.1, 说明我们模型对于这个例子很差,我们加入 gamma衰减.  衰减系数是. 0.9**2   如果我们分类特别好, 差只是0.1 那么我们进行衰减, 直接就是0.1**2   这个loss直接变小超级多. 也就是我们最后学习时候不太学这个分类了.从而能加速我们模型对于困难样本的学习速度!!!!!!!!!!!!!!!!!!!!!!!很重要.

        return loss


class YoloV4Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        super(YoloV4Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides

    def forward(
        self,
        p,
        p_d,
        label_sbbox,
        label_mbbox,
        label_lbbox,
        sbboxes,
        mbboxes,
        lbboxes,
    ):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides
# 分3种步长进行操作.
        (
            loss_s,
            loss_s_ciou,
            loss_s_conf,
            loss_s_cls,
        ) = self.__cal_loss_per_layer(
            p[0], p_d[0], label_sbbox, sbboxes, strides[0]
        )
        (
            loss_m,
            loss_m_ciou,
            loss_m_conf,
            loss_m_cls,
        ) = self.__cal_loss_per_layer(
            p[1], p_d[1], label_mbbox, mbboxes, strides[1]
        )
        (
            loss_l,
            loss_l_ciou,
            loss_l_conf,
            loss_l_cls,
        ) = self.__cal_loss_per_layer(
            p[2], p_d[2], label_lbbox, lbboxes, strides[2]
        )

        loss = loss_l + loss_m + loss_s
        loss_ciou = loss_s_ciou + loss_m_ciou + loss_l_ciou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_ciou, loss_conf, loss_cls

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")  # 这部分代码要参考yolo_head.py :21 行 中的p的物理含义来进行分析. #我们首先看bce源码,知道他是自己算log的. 输入的是 一堆数字即可,sigmoid之前的.
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none") # focalloss 源码也一样, 看出来也是sigmoid之前的.然后我们看yolo_head:21行. 知道p_d里面的概率是sigmoid之后的. 所以我们对于bce和FOCAL 函数传入的一定是p的才行.

        batch_size, grid = p.shape[:2]
        img_size = stride * grid  # stride表示每一个grid的pixel大小.

        p_conf = p[..., 4:5]       # 这2个要sigmoid之前的.
        p_cls = p[..., 5:]        # 这2个要sigmoid之前的.

        p_d_xywh = p_d[..., :4] # 只需要这个一个是绝对坐标.

        label_xywh = label[..., :4]        # 下面几个就简单了,就是ground true呗,直接抽取即可.
        label_obj_mask = label[..., 4:5] # 还需要参考datasets.py代码看看里面的label的物理含义.这地方物理含义就是 iou大于0.3了才是1.否则是0.
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]

        # loss ciou=======计算2个box的     ciou:物理含义是.当前这个cell对应ground_true的位置一样的cell. 他们的ciou,交并比,那么这个box是否存在物体呢. ciou里面每一个cell都有计算. 而实际上我们只学习那些cell中存在box的即可.所以下面需要算  label_obj_mask=========我们整个模型不用管 非box预测到哪, 只需要管存在box 的cell我们预测准即可. 因为后面有置信度跟着呢.label_mix
        ciou = tools.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[
            ..., 3:4
        ] / (img_size ** 2) # 面积系数.物体越大系数越小. 所以给小物体检测加了更大权重.让他们效果更好.======这个部分是提供loss衰减的.
        loss_ciou = label_obj_mask * bbox_loss_scale * (1.0 - ciou) * label_mix
 # 只计算有物体部分的iou======因为iou越大说明越好,所以1-ciou表示loss.     label_obj_mask 用来只计算带物体的地方做loss,不带物体的loss没意义.=======物理含义是这样的.==========我们只学这个cell里面带物体的. 这样可以降低超多计算量.
        # loss confidence=========下面是计算置信度的.
        iou = tools.CIOU_xywh_torch( # p_d_xywh 是 计算出来的box 结果.算ciou
            p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        ) # 计算每一个anchor和  150个箱子互相的iou值.
        iou_max = iou.max(-1, keepdim=True)[0]   # 看重合度最大的那个ioc值,作为这个anchor的ioc值. 表示这个anchor的置信度!!!!!!!!!!!!!!!!!!!!!
        label_noobj_mask = (1.0 - label_obj_mask) * (
            iou_max < self.__iou_threshold_loss
        ).float()         #小于阈值的都进行mask.        noobj_maks计算方法就是 (1.0 - label_obj_mask) 表示anchor 1表示没有物体, 然后 我们预测值也都小于阈值. 所以这个值就是我们预测这个cell不是物体,并且这个cell真实情况也不是物体.这种情况我们写1.
#============唯一一行体会感觉不深的.
        loss_conf = (
            label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) # 注意这个focal 用的是 reduction=none ,  就是输入什么shape,输出什么shape,因为他只是一个置信度. 输入的不是一个分布,而是一个0-1之间的float概率值. 结果就是置信度p_conf距离真实标签的距离.
            + label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)
        ) * label_mix  # 看第一个分量,表示在真正存在物体的anchor上我们计算 我们是物体的概率和真实物体的概率的 focal loss,第二个分量比较复杂了. 计算的是我们预测了不是物体, 真实情况这个anchor也真不是物体, 这就是负样本, 这里面focal也照样计算loss. 把这些加起来即可. 还是跟之前思路一样,我们只关心预测正确的东西的loss, 预测错了的不用管, 预测错包括对的预测错了.和错的预测对了.原因也是这么算可以大幅度降低运算量.======总共就是少了一种情况我们忽略他的loss, 就是 phat预测为1, p真实 是0的情况. 这种为什么不要修正呢, 我理解因为就算预测为了box,问题也不大,因为他的真实box没有.还是会收敛到0长度的box          而 phat预测为0, p 预测为1,这种是严重的错误,肯定需要学.===================================这个地方这么解释感觉还是有点牵强, 需要多理解, 以后多看看这个地方的处理!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # loss classes
        loss_cls = (
            label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix
        )

        loss_ciou = (torch.sum(loss_ciou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_ciou + loss_conf + loss_cls

        return loss, loss_ciou, loss_conf, loss_cls


if __name__ == "__main__":
    from model.build_model import Yolov4

    net = Yolov4()

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3, 52, 52, 3, 26)
    label_mbbox = torch.rand(3, 26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3, 26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    loss, loss_xywh, loss_conf, loss_cls = YoloV4Loss(
        cfg.MODEL["ANCHORS"], cfg.MODEL["STRIDES"]
    )(p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)
