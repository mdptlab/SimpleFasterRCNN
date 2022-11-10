import numpy as np

from anchor_box_utils import anchor_locs_to_center_format, center_format_to_corner_coordinates_format, \
    corner_coordinates_format_to_center_format, get_anchor_locations, find_ious


# TODO: Check again why x, y, h, w format in this case?
def anchor_locs_to_roi(anchor_locs, anc_height, anc_width, anc_ctr_y, anc_ctr_x):
    """
    anchor_locs is a numpy data for anchor_locs predicted for a SINGLE image
    """
    ctr_y, ctr_x, h, w = anchor_locs_to_center_format(anchor_locs, anc_height, anc_width, anc_ctr_y, anc_ctr_x)
    return center_format_to_corner_coordinates_format(ctr_x, ctr_y, h, w)


def anchors_and_anchor_locs_to_roi(anchors, anchor_locs):
    """
    anchor_locs is a numpy data for anchor_locs predicted for a SINGLE image
    Convert the dy, dx, dh, dw format to box format for the regions of interest
    """
    anc_height, anc_width, anc_ctr_y, anc_ctr_x = corner_coordinates_format_to_center_format(anchors)
    roi = anchor_locs_to_roi(anchor_locs, anc_height, anc_width, anc_ctr_y, anc_ctr_x)
    return roi


def generate_proposals(image_idx, anchors, pred_anchor_locs, objectness_score, is_training,
                       nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000,
                       n_test_post_nms=300, min_size=16, img_size=(800, 800)):
    """
    The proposal function takes the following parameters
    - is_training: whether it is training mode or testing mode
    - n_train_pre_nms - number of bounding boxes before non-maximum supression during training
    - n_train_post_nms - number of bounding boxes after non-maximum supression during training
    - n_test_pre_nms - number of bounding boxes before non-maximum supression during testing
    - n_test_post_nms - number of bounding boxes after non-maximum supression during testing
    - min_size - minimum height of the object required to create a proposal

    Faster R-CNN says, RPN proposals highly overlap with each other.
    To reduce redunddancy, we adopt non-maximum supression (NMS) on the proposal regions based on their class scores.
    We fix the IoU threshold for NMS at 0.7, which leaves us about 2000 proposal regions per image.
    After an ablation stdy, the authors show that NMS does not harm the ultimate detection accuracy, but substantially
    reduces the number of proposals.
    After NMS, we use the top-N ranked proposal regions for detection. In the following we train Fast R-CNN
     using 2000 RPN proposals.
    During the testing they evaluate only 300 proposals, they have tested this with various numbers and oob tainde this.

    We do need to do the following  things to generate region of interest proposals to the network
    1. Convert the loc predictions from rpn network to bounding box [y1, x1, y2, x2]
    2. Clip the predicted boxes to the image
    3. Remove predicted boxes with either height or width < threshold (min_size)
    4. Sort all (proposal, score) pairs by score from highest to lowest
    5. Take top pre-nms_topN (e.g., 12000 while training and 6000 while testing)
    6. Apply nms threshold > 0.7
    7. Take top pos_nms_topN (e.g., 2000  while training and 300 while testing)

    """

    pred_anchor_locs_numpy = pred_anchor_locs[image_idx].to("cpu").data.numpy()
    objectness_score_numpy = objectness_score[image_idx].to("cpu").data.numpy()
    # Generate all region of interests from anchors (22500) and predicted anchor locations (numpy)
    roi = anchors_and_anchor_locs_to_roi(anchors, pred_anchor_locs_numpy)
    # Clip the predicted boxes to the image (which larger or negative then clip to othe image size)
    if not isinstance(img_size, tuple):
        img_size = (img_size, img_size)

    # roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
    # roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

    # Remove predicted boxes with either height or width < threshold
    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    roi = roi[keep, :]
    # Also filter the objectness_score
    score = objectness_score_numpy[keep]

    # Sort all (proposal, score) pairs by score from highest to lowest
    order = score.ravel().argsort()[::-1]
    # Take the top pre_nms_topN (e.g., 12000 while training and 6000 while testing)
    n_pre_nms = n_train_pre_nms if is_training else n_test_pre_nms
    order = order[:n_pre_nms]
    roi = roi[order, :]
    # Also take the toop scores only
    score = score[order]

    n_post_nms = n_train_post_nms if is_training else n_test_post_nms
    # Split the locations into y1, x1, y2, x2
    y1 = roi[:, 0]
    x1 = roi[:, 1]
    y2 = roi[:, 2]
    x2 = roi[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # This is the index (argsort) for the new array (different index already, after filtered)
    order = score.argsort()[::-1] # this is to take descending order start:stop:step

    # Only keep those who have overlap less than a threshold with the top N and we do from top down
    keep = []
    while order.size > 0:
        i = order[0]  # take the 1st elt in order and append to keep
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (area[i] + area[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        # +1 here because the order array in the processing is order[1:], so it was cut off 1 compared to original order
        order = order[inds + 1]
    keep = keep[:n_post_nms]  # while training/testing , use accordingly
    roi = roi[keep]  # the final region proposals
    if (len(roi) == 0):
        print(roi)
    return roi


def generate_proposal_targets(image_idx, anchors, pred_anchor_locs, objectness_score, is_training, nms_thresh, n_train_pre_nms,
                              n_train_post_nms, n_test_pre_nms, n_test_post_nms, min_size, img_size, bbox, labels,
                              n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_lo=0.0,
                              neg_iou_thresh_hi=0.5):
    """
    1. For each roi,  find IoU with all other ground truth [N, n]
    2. Find which ground truth has highest iou with the roi [N], will be the class labels for each and every proposal
    3. Take the roi with iou >= pos_iou_thresh to be positive  sample
    4. If number of positive samples > pos_ratio * n_sample (32 in this case), sample it
    5. If IoU is between [neg_iou_thresh_lo, neg_iou_thresh_hi] then assign negative label 0 to the region proposal
    6. neg_samples: we  randomly 128 - number of pos region proposals on this image and assign  to 0 to these proposals
    7. Convert the locations of ground-truth objects for each region proposal to the required format
    8. Output the labels and locations for the sampled rois
    Inputs:
    - roi: region of interest generated from RPN (anchors + anchor_locs)
    - bbox: the ground-truth bounding boxes
    - labels: the label (1, 2, 3...) for the classes
    - n_sample - the number of roi to be sampled (default is 128)
    - pos_ratio - the ratios of positive anchors (out of the n_sample default is 0.25, i.e., 32 sample rois in this case)
    - pos_iou_thresh - the threshold to consider a roi is positive (using interaction over union)
    - neg_iou_thresh_hi - the threshold (upper bound) to consider a roi is negative
    - neg_iou_thresh_lo - the threshold (lower bound) to consider a roi is negative

    """
    if not isinstance(img_size, tuple):
        img_size = (img_size, img_size)
    roi = generate_proposals(image_idx, anchors, pred_anchor_locs, objectness_score, is_training, nms_thresh, n_train_pre_nms,
                             n_train_post_nms, n_test_pre_nms, n_test_post_nms, min_size, img_size)
    # find the ious for each ground truth object with the region proposals
    iou = find_ious(roi, bbox)
    # find out which ground truth has high IoU for each region proposals, and their correspoding iou value.
    gt_assignment = iou.argmax(axis=1)
    max_iou = iou.max(axis=1)
    # Assign the label for each proposal
    gt_roi_label = labels[gt_assignment]
    # Select the foreground rois as per pos_iou_thresh.
    # We also sample n_sample x pos_ratio (128 x 0.25 = 32) foreground samples.
    # So in case if we get less than 32 positive samples we will leave it as it is,
    # in case we get more than 32 foreground samples, we will sample 32 samples from the positive samples
    pos_roi_per_image = pos_ratio * n_sample  # 32 in this case
    pos_index = np.where(max_iou >= pos_iou_thresh)[0]
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    if pos_index.size > pos_roi_per_image:
        pos_index = np.random.choice(
            pos_index, size=pos_roi_per_image, replace=False
        )
    # Similarly we do for negative (background) region proposals
    # also, if we have region proposals with IoU between neg_iou_thresh_lo and neg_iou_thresh_hi
    # for the ground truth object assigned to it earlier, we assign 0 label to the region proposal.
    # We will sample n (n_sample - pos_samples, 128 - 32 = 96) region proposals from these negative samples.
    neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
    if neg_index.size > neg_roi_per_this_image:
        neg_index = np.random.choice(
            neg_index, size=neg_roi_per_this_image, replace=False
        )
    # gather positive samples and negative samples
    # TODO: Can we do something smarter than sampling? Or sample with probabilities based on score?
    keep_index = np.append(pos_index, neg_index)
    # gt_roi_labels were simply assigned to their corresponding max iou without considering actual iou value
    # (using threshold + samples)
    gt_roi_labels = gt_roi_label[keep_index]
    # so now assign background for those who are negative (small overlap + not in sampled value)
    gt_roi_labels[pos_roi_per_this_image:] = 0
    sample_roi = roi[keep_index]
    # pick the ground truth objects for these sample_roi and later parameterize
    # as we have done while assigning locations to anchor boxes in section 2
    bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]  # ground truth
    gt_roi_locs = get_anchor_locations(sample_roi, bbox_for_sampled_roi)

    return sample_roi, gt_roi_locs, gt_roi_labels
