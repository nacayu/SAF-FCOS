"""
This file contains specific functions for computing losses of FCOS
file
"""

import os

import torch
from torch import nn

from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        Get points mask whether points inside any 2d bboxes
        FCOS 的 BBox Assigner 过程是通过 regress_ranges 和 center_sample_radius 两个参数控制
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs) 
        # all gt bboxes assigned to every location
        gt = gt[None].expand(K, num_gts, 4)     
        # get gt center
        center_x = (gt[..., 0] + gt[..., 2]) / 2 
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p   
            # 例如 radius=1.5，对于 stride=4 的层，采样半径是 1.5x4, 对于 stride=8 的层，采样半径是 1.5x8
            stride = strides[level] * radius
            # get stride area range for points
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region, compare radius area and gt bboxes area range(get Intersection area between gt and center_sampling)
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        # 每个points到所有gt bboxes四个边的距离: l, r, t, b
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        
        center_bbox = torch.stack((left, top, right, bottom), -1)
        # True if the corresponding points in the feature map lies inside any center sampling area
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        """
        Get  per points labels and regression gt
        Arguments:
        points: list([tensor])
        targets: list([BoxList])
        Returns:
        labels_level_first: list([tensor]) features_level * n
        reg_targets_level_first: list([tensor]) features_level * n
        """
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        # 不同尺度特征层分配不同大小的object size范围: feature map越小分配size越大
        for l, points_per_level in enumerate(points):
            # 五个size按照feature level分别分配
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            # 每个level对应的所有points都分配对应object size相应模板尺寸
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
        # get all points level together
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        
        # 基于center_sampling和 regress_ranges原则得到每个点样本类别
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )
        # 将每个图片的label按照level分割: 便于encoder中除以stride
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
        # get level-first labels
        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)
            
            # encoder
            # 最终计算的reg_targets需要除以对应level的strides
            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        """
         基于 center_sampling 和 regress_ranges采样正负样本:
         - 如果不开启中心采样，那么 gt bbox 内部的所有点都属于正样本
         - 一旦开启了中心采样，则需要依据采样半径计算出每个 gt bbox 内部哪些点属于正样本
         - 基于center_sampling 没有考虑 FPN 的按照 scale 分配样本的特性，将所有 gt bbox 都广播到每个输出层
         - 基于 regress_ranges 通过 regree_ranges 进一步按照 scale 原则进行区分
         
         正样本要符合两个条件：
         1. inside 2d bboxes and center sampling area
         2. inside regression range
        """
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        # get labels for images separately
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()
            # 计算原图上center points和当前图片所有 gt bbox 4 条边的距离
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            # get mask for points whether inside center sampling, shape: [num_points, num_gt_bboxes]
            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            # max offset from center points to all 2d bboxes border
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            # mask locations: 1. points not in center_sampling area     2. points regression attrs don't meet object template sizes
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            # if there are still more than one objects for a location,
            # assign center points with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            # get locations' gt reg
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        # get per points labels and regression gt (level based)
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        # reshape for easily calculating losses
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
        # cat along on  feature levels
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        # ignore labels == 0
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # select box, reg, center-ness
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        
        """
        cls_loss: sigmoid focal loss
        reg_loss: IoU loss with centerness
        centerness_loss: BCE loss
        """
        # calculate for all boxes
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu
        # calculate for positive boxes
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
