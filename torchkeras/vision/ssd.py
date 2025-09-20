from math import sqrt
import itertools
import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List
from torch import nn, Tensor
import numpy as np
from .resnet import resnet50
import random 

def box_area(boxes):
    """
    bounding boxes type: (x1, y1, x2, y2)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou_tensor(boxes1, boxes2):
    """
    bounding boxes type: (x1, y1, x2, y2)
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# This function is from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-src
        Transform between (bboxes, lables) <-> SSD output
        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format
        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)  # default boxes的数量
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        """
        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes
        """
        # [nboxes, 8732]
        self.dboxes = self.dboxes.to(bboxes_in.device)
        ious = calc_iou_tensor(bboxes_in, self.dboxes)  # 计算每个GT与default box的iou
        # [8732,]
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)  # 寻找每个default box匹配到的最大IoU
        # [nboxes,]
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)  # 寻找每个GT匹配到的最大IoU

        # 将每个GT匹配到的最佳default box设置为正样本（对应论文中Matching strategy的第一条）
        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)  # dim, index, value
        # 将相应default box匹配最大IOU的GT索引进行替换
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64).to(bboxes_in.device)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        # 寻找与GT iou大于0.5的default box,对应论文中Matching strategy的第二条(这里包括了第一条匹配到的信息)
        masks = best_dbox_ious > criteria
        # [8732,]
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64).to(bboxes_in.device)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        # 将default box匹配到正样本的位置设置成对应GT的box信息
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        # Transform format to xywh format
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])  # x
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])  # y
        w = bboxes_out[:, 2] - bboxes_out[:, 0]  # w
        h = bboxes_out[:, 3] - bboxes_out[:, 1]  # h
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            将box格式从xywh转换回ltrb, 将预测目标score通过softmax处理
            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox
            bboxes_in: 是网络预测的xywh回归参数
            scores_in: 是预测的每个default box的各目标概率
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        # Returns a view of the original tensor with its dimensions permuted.
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]   # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]   # 预测的w, h回归参数

        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        # 将box格式从xywh转换回ltrb（方便后面非极大值抑制时求iou）, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = []
        # 遍历一个batch中的每张image数据
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single(bbox, prob, criteria, max_output))
        return outputs

    def decode_single(self, bboxes_in, scores_in, criteria, num_output=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # 移除归为背景类别的概率信息
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        # 移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size   # 输入网络的图像大小 300
        # [38, 19, 10, 5, 3, 1]
        self.feat_size = feat_size  # 每个预测层的feature map尺寸

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        # [8, 16, 32, 64, 100, 300]
        self.steps = steps    # 每个特征层上的一个cell在原图上的跨度

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales  # 每个特征层上预测的default box的scale

        fk = fig_size / np.array(steps)     # 计算每层特征层的fk
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios  # 每个预测特征层上预测的default box的ratios

        self.default_boxes = []
        # size of feature and number of feature
        # 遍历每层特征层，计算default box
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size  # scale转为相对值[0-1]
            sk2 = scales[idx + 1] / fig_size  # scale转为相对值[0-1]
            sk3 = sqrt(sk1 * sk2)
            # 先添加两个1:1比例的default box宽和高
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # 再将剩下不同比例的default box宽和高添加到all_sizes中
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # 计算当前特征层对应原图上的所有default box
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):  # i -> 行（y）， j -> 列（x）
                    # 计算每个default box的中心坐标（范围是在0-1之间）
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        # 将default_boxes转为tensor格式
        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)  # 这里不转类型会报错
        self.dboxes.clamp_(min=0, max=1)  # 将坐标（x, y, w, h）都限制在0-1之间

        # For IoU calculation
        # ltrb is left top coordinate and right bottom coordinate
        # 将(x, y, w, h)转换成(xmin, ymin, xmax, ymax)，方便后续计算IoU(匹配正负样本时)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]   # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]   # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]   # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]   # ymax

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order='ltrb'):
        # 根据需求返回对应格式的default box
        if order == 'ltrb':
            return self.dboxes_ltrb

        if order == 'xywh':
            return self.dboxes


def dboxes300_coco():
    figsize = 300  # 输入网络的图像大小
    feat_size = [38, 19, 10, 5, 3, 1]   # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]   # 每个特征层上的一个cell在原图上的跨度
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]  # 每个特征层上预测的default box的scale
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # 每个预测特征层上预测的default box的ratios
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).
    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.
    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU < iou_threshold
    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold
    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    # 为每一个类别生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4]
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0),
                                        requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5
        self.max_output = 100

    def scale_back_batch(self, bboxes_in, scores_in):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """
            1）通过预测的boxes回归参数得到最终预测坐标
            2）将box格式从xywh转换回ltrb
            3）将预测目标score通过softmax处理
            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox
            bboxes_in: [N, 4, 8732]是网络预测的xywh回归参数
            scores_in: [N, label_num, 8732]是预测的每个default box的各目标概率
        """

        # Returns a view of the original tensor with its dimensions permuted.
        # [batch, 4, 8732] -> [batch, 8732, 4]
        bboxes_in = bboxes_in.permute(0, 2, 1)
        # [batch, label_num, 8732] -> [batch, 8732, label_num]
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]   # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]   # 预测的w, h回归参数

        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        # scores_in: [batch, 8732, label_num]
        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_single(self, bboxes_in, scores_in, criteria, num_output):
        # type: (Tensor, Tensor, float, int) -> Tuple[Tensor, Tensor, Tensor]
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        # [num_classes] -> [8732, num_classes]
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # 移除归为背景类别的概率信息
        bboxes_in = bboxes_in[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
        scores_in = scores_in[:, 1:]  # [8732, 21] -> [8732, 20]
        labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
        scores_in = scores_in.reshape(-1)  # [8732, 20] -> [8732x20]
        labels = labels.reshape(-1)  # [8732, 20] -> [8732x20]

        # remove low scoring boxes
        # 移除低概率目标，self.scores_thresh=0.05
        # inds = torch.nonzero(scores_in > 0.05).squeeze(1)
        inds = torch.where(torch.gt(scores_in, 0.05))[0]
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)
        # keep = keep.nonzero().squeeze(1)
        keep = torch.where(keep)[0]
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    def forward(self, bboxes_in, scores_in):
        # 通过预测的boxes回归参数得到最终预测坐标, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        # 遍历一个batch中的每张image数据
        # bboxes: [batch, 8732, 4]
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):  # split_size, split_dim
            # bbox: [1, 8732, 4]
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single(bbox, prob, self.criteria, self.max_output))
        return outputs
    
    
class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        # 修改conv4_block1的步距，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, loc):
        # type: (Tensor) -> Tensor
        """
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        :param loc: anchor匹配到的对应GTBOX Nx4x8732
        :return:
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels
            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        # 获取正样本的mask  Tensor: [N, 8732]
        mask = torch.gt(glabel, 0)  # (gt: >)
        # mask1 = torch.nonzero(glabel)
        # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)

        # sum on four coordinates, and mask
        # 计算定位损失(只有正样本)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]

        # hard negative mining Tenosr: [N, 8732]
        con = self.confidence_loss(plabel, glabel)

        # positive mask will never selected
        # 获取负样本
        con_neg = con.clone()
        con_neg[mask] = 0.0
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙

        # number of negative three times positive
        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor [N]

        # avoid no object detected
        # 避免出现图像中没有GTBOX的情况
        total_loss = loc_loss + con_loss
        # eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
        return ret

    
class SSD300(nn.Module):
    def __init__(self, num_classes=21,backbone=None):
        super(SSD300, self).__init__()
        if backbone is None:
            backbone = Backbone()
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")
        self.feature_extractor = backbone

        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = []
        confidence_extractors = []

        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        default_box = dboxes300_coco()
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

    def _build_additional_features(self, input_size):
        """
        为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
        :param input_size:
        :return:
        """
        additional_blocks = []
        # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):
        x = self.feature_extractor(image)

        # Feature Map 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        detection_features = torch.jit.annotate(List[Tensor], [])  # [x]
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732
        
        result = {'loc': locs, 'conf':confs}
        if targets is not None:
            #encode
            for t in targets:
                boxes = t['boxes']
                labels = t['labels']
                bboxes_out, labels_out = self.encoder.encode(boxes, labels)
                t['boxes'],t['labels'] = bboxes_out, labels_out

            #stack
            boxes = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]
            bboxes_out = torch.stack(boxes, dim=0)
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            labels_out = torch.stack(labels, dim=0)

            # ploc, plabel, gloc, glabel
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            result['loss'] = loss
        return result 
    
    @torch.no_grad()
    def predict(self, image):
        self.eval()
        out = self.forward(image)
        locs, confs = out['loc'], out['conf']
        # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        # results = self.encoder.decode_batch(locs, confs)
        results = self.postprocess(locs, confs)
        predictions = []
        for result in results:
            boxes,labels,scores = result
            predictions.append({'boxes':boxes,'labels':labels,'scores':scores})
        return predictions
    