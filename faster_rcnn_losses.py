import numpy as np
import torch
import torch.nn.functional as F


def calculate_classification_loss(score, gt_score):
    # ignore the labels with target = -1 (invalid ones, or those which are neither positive or negative)
    cls_loss = F.cross_entropy(score, gt_score.long(), ignore_index=-1)
    return cls_loss


def calculate_regression_loss(gt_score, loc, gt_loc):
    # calculate losses only on positive gt_score
    pos = gt_score > 0
    mask = pos.unsqueeze(1).expand_as(loc)
    mask_loc_preds = loc[mask].view(-1, 4)
    mask_loc_targets = gt_loc[mask].view(-1, 4)
    # the  loss
    x = torch.abs(mask_loc_targets - mask_loc_preds)
    loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))

    return loc_loss


def calculate_loss(score, gt_score, loc, gt_loc, loc_loss_lambda=10.):
    # The classification loss
    cls_loss = calculate_classification_loss(score, gt_score)
    # The location regression loss
    loc_loss = calculate_regression_loss(gt_score, loc, gt_loc)

    n_reg = (gt_score > 0).float().sum()
    loc_loss = loc_loss.sum() / n_reg
    loss = cls_loss + (loc_loss_lambda * loc_loss)
    return loss


# RPN Loss
def calculate_rpn_loss(anchor_locations, anchor_labels, pred_anchor_locs, pred_cls_scores, device):
    """

    :param anchor_locations: numpy anchor locations taken from RPN section
    :param anchor_labels: numpy anchor labels taken from RPN section
    :param pred_anchor_locs: tensor the predicted anchor locations taken from RPN section
    :param pred_cls_scores: tensor the predicted anchor class scores taken from RPN section
    :return:
    """
    rpn_loc = pred_anchor_locs[0]
    rpn_score = pred_cls_scores[0]
    gt_rpn_loc = torch.from_numpy(anchor_locations).to(device)
    gt_rpn_score = torch.from_numpy(anchor_labels).to(device)

    rpn_loss = calculate_loss(rpn_score, gt_rpn_score, rpn_loc, gt_rpn_loc)
    return rpn_loss


# Fast RCNN Loss
def calculate_fast_rcnn_loss(roi_cls_loc, roi_cls_score, gt_roi_locs, gt_roi_labels, device):
    """

    :param roi_cls_loc: tensor generated from Fast RCNN
    :param roi_cls_score: tensor generated from Fast RCNN
    :param gt_roi_locs: tensor generated from Fast RCNN
    :param gt_roi_labels: tensor generated from Fast RCNN
    :return:
    """
    # Converting ground truth to torch variable
    gt_roi_loc = torch.from_numpy(gt_roi_locs).to(device)
    gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long().to(device)

    # location data
    n_sample = roi_cls_loc.shape[0]
    roi_loc = roi_cls_loc.view(n_sample, -1, 4)
    roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
    roi_loss = calculate_loss(roi_cls_score, gt_roi_label, roi_loc, gt_roi_loc)
    return roi_loss


def calculate_faster_rcnn_loss(anchor_locations, anchor_labels, pred_anchor_locs, pred_cls_scores, roi_cls_loc,
                               roi_cls_score, gt_roi_locs, gt_roi_labels, device):
    rpn_loss = calculate_rpn_loss(anchor_locations, anchor_labels, pred_anchor_locs, pred_cls_scores, device)
    roi_loss = calculate_fast_rcnn_loss(roi_cls_loc, roi_cls_score, gt_roi_locs, gt_roi_labels, device)
    total_loss = rpn_loss + roi_loss
    return total_loss
