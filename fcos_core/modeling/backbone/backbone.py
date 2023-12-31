# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from fcos_core.modeling import registry
from fcos_core.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet

"""
这部分代码是使用 registry.BACKBONES.register 注册函数的装饰器语法，用于将函数注册为指定名称的骨干网络。

具体来说，@registry.BACKBONES.register("R-50-FPN") 将下面定义的 build_resnet_fpn_backbone 函数注册为名称为 "R-50-FPN" 的骨干网络。
同样地，@registry.BACKBONES.register("R-101-FPN") 将下面定义的 build_resnet_fpn_backbone 函数注册为名称为 "R-101-FPN" 的骨干网络。

通过使用这种注册机制，可以在后续的代码中根据名称来选择和配置相应的骨干网络模型。例如，在配置文件中指定使用 "R-50-FPN" 的骨干网络，
然后通过 registry.BACKBONES["R-50-FPN"] 来获取相应的模型构建函数。这样可以方便地切换和配置不同的骨干网络模型。
"""

@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    # specified ResNet added radar branch and fusion branch
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    # 
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    # connect resnet and fpn
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    # 调用cfg.MODEL.BACKBONE.CONV_BODY对应的网络并输入cfg
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
