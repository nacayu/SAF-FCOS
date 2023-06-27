import torch

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import remove_small_boxes


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(
            self,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            num_classes,
            bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        # 在NMS前先根据分类得分进行筛选，选取大于0.05的box
        # (N, H*W, C) bool True or False
        candidate_inds = box_cls > self.pre_nms_thresh
        # [N, ], 每个point类别未知, pre_nms_top_n[i]为point_i的所有类别分数 > score的个数 * N个图片的points数量, 一个特征点位置可能在多个类别上都成为候选
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # [N, ] 限制候选目标数量，默认最多是1000个
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]
        
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            # [per_candidate_inds.sum(), 2]: index 0 indicate points position, index 1 indicate class 
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            #  points点索引: (per_candidate_inds.sum(), )
            per_box_loc = per_candidate_nonzeros[:, 0]
            # +1是因为网络结构中，在分类预测的卷积层输出通道数里并未包含背景类，因此这里将0留给背景
            per_class = per_candidate_nonzeros[:, 1] + 1
            # points回归的预测结果: [H*W,4]
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            # points位置
            per_locations = locations[per_box_loc]
            # 
            per_pre_nms_top_n = pre_nms_top_n[i]
            # 若当前候选目标(score > 0.05)数量(包括一个点在多个类别上成为候选)超过了上限值则进行截断
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            # 由回归的4个量计算出bbox两个对角的坐标x1y1x2y2(对应到输入图像)
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)
            # 实例化BoxList()对象
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            # 将bbox(四个角坐标)限制在输入图像尺寸范围内
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # self.min_size默认为0
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        # locations, box_cls, box_regression, centerness的长度都等于特征层数
        # 而image_sizes的长度等于一个batch中的图片数量
        # 注意，返回的每个预测结果封装在BoxList()实例中，通过这个实例的属性可以获取
        # 对应的bbox、类别以及分数
        
        # 依次处理各个特征层的预测结果
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            # 返回list，代表各图在单个特征层的预测结果
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )
        # get per images boxlists from multi-level(Re-group)
        boxlists = list(zip(*sampled_boxes))
        # 将一张图中所有特征层的预测结果拼接在一起，使得每张图的预测结果就是1个BoxList实例
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        # 依次处理各张图像
        for i in range(num_images):
            # multiclass nms
            # NMS后返回的还是BoxList()实例
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            # NMS后剩余的目标数量
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            # 若经过NMS后该图的目标数量超出了上限(默认100)，则进行截断
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                # 得分已排好序
                cls_scores = result.get_field("scores")
                # 取排在超出数量限制的那个点对应的得分作为阀值
                # 这样就能限制目标数量，最终获取得分最高的前100个
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            results.append(result)

        return results

def make_fcos_postprocessor(config):
    # 默认0.05 若一个预测结果在分类得分上连0.05都不足则剔除掉
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    # 默认1000 NMS前每张图仅保留得分最高的前1000个预测结果
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    # 默认0.6 NMS阀值，0.6以上则视为重叠框
    nms_thresh = config.MODEL.FCOS.NMS_TH
    # 默认100 NMS后每张图片最多保留多少个目标
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    # 是否采用测试增强(TTA) 默认False
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
