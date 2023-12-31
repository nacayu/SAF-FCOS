# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torch import nn

from fcos_core.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads
from ..rpn.rpn import build_rpn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        # get multi-level fusion features
        self.backbone = build_backbone(cfg)
        # get first stage RPN proposals
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # two-stage method: extract proposal features to refine
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, pc_images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            pc_images (list[Tensor] or PCImageList): pc_images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        pc_images = to_image_list(pc_images)
        features = self.backbone((images.tensors, pc_images.tensors))
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
