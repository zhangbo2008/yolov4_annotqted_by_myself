import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)
# bs,anchor数量, 5+分类数量, 格子行,格子列.
        p_de = self.__decode(p.clone()) #研究一下这个decode是干啥呢????????======知道了原来p这个东西出来的都是box内部的相对坐标.我们要把他decode成我们label对应的绝对坐标才能计算loss.所以p_de才是真正的输出.

        return (p, p_de)
#============================================================查询p的含义!!!!!!!!!!!!!!!!!!!!2021-05-18,16点54+++++++++++++++++++++++非常重要!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!对于loss的理解.
    def __decode(self, p):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)
#下面4行特别关键,赋予了我们整个yolo网络的物理含义. p的0,1表示  预测的中心距离cell内部的中心点的偏移量.看44行一下就懂了.
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4] #
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = (
            grid_xy.unsqueeze(0)
            .unsqueeze(3)
            .repeat(batch_size, 1, 1, 3, 1)
            .float()
            .to(device)
        )
#========具体含义直接看这里面吧. 从这里面可以看懂p的物理含义.
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride  #         pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride  # 保证了 dx dy 一定在当前cell里面.
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride # 保证 w,h torch.exp(conv_raw_dwdh) 一定大于0.
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return (
            pred_bbox.view(-1, 5 + self.__nC)
            if not self.training
            else pred_bbox
        )
