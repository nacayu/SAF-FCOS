import math

import torch
import torch.nn.functional as F
from torch import nn

from fcos_core.layers import DFConv2d
from fcos_core.layers import Scale
from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        # num_classes: 2-1 -> 1
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        
        # num convs: 4        
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            
            # cls: [conv, groupnorm, relu]
            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            # bbox: [conv, groupnorm, relu]
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        
        # get cls, bbox, centerness prediction
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        # shared Decoder layer: get preds from fpn features x(tuple)
        for l, feature in enumerate(x):
            # get cls, bbox features
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            # get class logits
            logits.append(self.cls_logits(cls_tower))
            # get centerness
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))
            # get bbox preds
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))       # self.scales: box_tower * self.scale
            if self.norm_reg_targets:
                # bbox_pred: >= 0
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        
        box_cls, box_regression, centerness = self.head(features)
        # 输出是长度为 5 的 list，每个 item 都是存储该特征图点映射到原图上的坐标点 y,x 值
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        # box_regression在test时，head输出的结果已经预先进行stride相乘
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        """
            compute_locations_per_level: 先在特征图尺度计算所有坐标点，然后通过 stride 转化为原图尺度
        """
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
