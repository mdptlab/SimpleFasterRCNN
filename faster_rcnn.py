import torch
import torch.nn as nn

from anchor_box_utils import generate_anchors_for_bbox
from fastrcnn import adaptive_pooling, FastRCNN
from feature_extractor_utils import get_backbone
from roi_proposal_utils import generate_proposal_targets
from rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self, faster_rcnn_fe_extractor, image_size, sub_sample, mid_channels=512, in_channels=512, n_anchor=9,
                 n_classes=4, device=torch.device("cpu")):
        super(FasterRCNN, self).__init__()
        self.faster_rcnn_fe_extractor = faster_rcnn_fe_extractor.to(device)
        self.image_size = image_size
        self.sub_sample = sub_sample
        self.n_classes = n_classes
        self.fastrcnn = FastRCNN(num_of_classes=n_classes).to(device)
        self.rpn = RPN(mid_channels, in_channels, n_anchor).to(device)

    def forward(self, image, bbox, labels):
        # Get the settings
        image_size = self.image_size
        sub_sample = self.sub_sample
        out_map = self.faster_rcnn_fe_extractor(image)
        # if(torch.isnan(out_map).detach().cpu().numpy().sum()>0):
        #     print('There is nan')
        # 1. Generate anchor locations and their labels
        anchors, anchor_locations, anchor_labels = generate_anchors_for_bbox(image_size, sub_sample, bbox,
                                                                             pos_iou_threshold=0.7,
                                                                             neg_iou_threshold=0.3,
                                                                             pos_ratio=0.5,
                                                                             n_sample=256)

        # 2. Regional Proposal Network: generate anchor locations, objectness score, and their class scores
        pred_anchor_locs, objectness_score, pred_cls_scores = self.rpn(out_map)

        # 3. Generate the proposals (ROI)
        is_training = self.training
        nms_thresh = 0.7
        n_train_pre_nms = 12000
        n_train_post_nms = 2000
        n_test_pre_nms = 6000
        n_test_post_nms = 300
        min_size = 16
        img_size = (800, 800)
        image_idx = 0
        sample_roi, gt_roi_locs, gt_roi_labels = generate_proposal_targets(image_idx, anchors, pred_anchor_locs,
                                                                           objectness_score,
                                                                           is_training, nms_thresh, n_train_pre_nms,
                                                                           n_train_post_nms, n_test_pre_nms,
                                                                           n_test_post_nms, min_size, img_size, bbox,
                                                                           labels,
                                                                           n_sample=128, pos_ratio=0.5,
                                                                           pos_iou_thresh=0.5,
                                                                           neg_iou_thresh_lo=0.0,
                                                                           neg_iou_thresh_hi=0.5)
        if len(sample_roi) == 0:
            return
        # 4. Pass through adaptive pooling layer
        k = adaptive_pooling(out_map, sample_roi, image_idx=image_idx, sub_sample=sub_sample)

        # 5. Fast RCNN
        roi_cls_loc, roi_cls_score = self.fastrcnn(k)

        return anchor_locations, anchor_labels, pred_anchor_locs, pred_cls_scores, roi_cls_loc, roi_cls_score, \
               gt_roi_locs, gt_roi_labels
