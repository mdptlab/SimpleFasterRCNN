import functools

import numpy as np


@functools.lru_cache(maxsize=1)
def generate_anchor_centers(image_size, sub_sample):
    """
    Generate the center points (from original image) for every feature point. Feature is the one extracted by the base CNN model
    :param image_size: the size of the image (assumed to be squared so we will provide single value here)
    :param sub_sample: the number of sub-samples, e.g., the image is 800px and the extracted feature (base CNN) is 50, then the sub_sample = 16
    :return: the centers (inform of [y, x] coordinate for every  point in the feature map)
    """
    fe_size = image_size // sub_sample
    ctr_x = np.arange(sub_sample, (fe_size + 1) * sub_sample, sub_sample)
    ctr_y = np.arange(sub_sample, (fe_size + 1) * sub_sample, sub_sample)
    ctr = np.zeros((len(ctr_x) * len(ctr_y), 2))
    index = 0
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index, 1] = ctr_x[x] - sub_sample // 2
            ctr[index, 0] = ctr_y[y] - sub_sample // 2
            index += 1
    return ctr


# @functools.lru_cache(maxsize=1)
def generate_raw_anchors(image_size, sub_sample, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    Returns
    - anchors: list of anchors in form of y1, x1, y2, x2
    - valid_anchor_boxes: those are inside the boundary
    - index_inside: the indices of the boxes which are valid (inside the image)
    """
    centers = generate_anchor_centers(image_size, sub_sample)
    fe_size = image_size // sub_sample
    anchors_per_ctr = len(ratios) * len(anchor_scales)
    anchors = np.zeros(((fe_size * fe_size * anchors_per_ctr), 4))
    index = 0
    for c in centers:
        ctr_y, ctr_x = c
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
                w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])
                anchors[index, 0] = ctr_y - h / 2.
                anchors[index, 1] = ctr_x - w / 2.
                anchors[index, 2] = ctr_y + h / 2.
                anchors[index, 3] = ctr_x + w / 2.
                index += 1
    index_inside = np.where(
        (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= image_size) & (anchors[:, 3] <= image_size))[0]
    valid_anchor_boxes = anchors[index_inside]
    return anchors, valid_anchor_boxes, index_inside


# TODO: Should use numpy array to make these calculations more efficient
def find_ious(anchor_boxes, bbox):
    """
    Returns:
    Intersection over union of each anchor box with each of the bounding box the size is (number of anchor, number of ground-truth bounding boxes)
    """
    ious = np.empty((len(anchor_boxes), len(bbox)), dtype=np.float32)
    ious.fill(0)
    for num1, i in enumerate(anchor_boxes):
        ya1, xa1, ya2, xa2 = i
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        for num2, j in enumerate(bbox):
            yb1, xb1, yb2, xb2 = j
            box_area = (yb2 - yb1) * (xb2 - xb1)

            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])

            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = inter_area / (anchor_area + box_area - inter_area)
            else:
                iou = 0.
            ious[num1, num2] = iou
    return ious


def get_ious_info(image_size, sub_sample, bbox):
    """
    return three arrays
    - argmax_ious - tells which ground-truth object has max iou with each anchor
    - max_ious - tells the max_iou with groud-truth object for each anchor
    - gt_argmax_ious - tells the anchors with the highest IoU overlap with at least one of the ground-truth boxes
    """
    anchors, valid_anchor_boxes, index_inside = generate_raw_anchors(image_size, sub_sample)
    ious = find_ious(valid_anchor_boxes, bbox)
    # case 1
    # find the anchor indices which are closest to each of the ground truth box
    gt_argmax_ious = ious.argmax(axis=0)
    # the max ious with each of the ground truths
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    # case 2
    # for each of the anchors, it will  be related to which ground-truth box
    argmax_ious = ious.argmax(axis=1)
    # the corresponding iou values with the corresponding ground-truths it is attached to
    max_ious = ious[np.arange(len(index_inside)), argmax_ious]
    # find the anchor_boxes which have this max_ious (gt_max_ious)
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]
    return anchors, valid_anchor_boxes, index_inside, argmax_ious, max_ious, gt_argmax_ious, gt_max_ious


def generate_labels_for_valid_anchors(valid_anchor_boxes, gt_argmax_ious, max_ious, pos_iou_threshold,
                                      neg_iou_threshold):
    label = np.empty((len(valid_anchor_boxes),), dtype=np.int32)
    label.fill(-1)  # by default we ignore the anchor box (simply  valid ones)
    # Assign negative label (0) to all the anchor boxes which have max_iou less than negative threshold (c)
    label[max_ious < neg_iou_threshold] = 0
    # Assign positive label (1) to all the anchor boxes which have highest IoU overlap with a ground-truth box (a)
    label[gt_argmax_ious] = 1
    # Assign positive label (1) to all the anchor boxes which have max_iou grater than positive threshold (b)
    label[max_ious >= pos_iou_threshold] = 1
    return label


def sample_anchors(label, pos_ratio=0.5, n_sample=256):
    """
    set label to 1 as to be positive
    set label to 0 to be negative
    set label to -1 as to be ignored
    """
    # total positive samples
    n_pos = int(pos_ratio * n_sample)
    # Now we need to randomly sample n_pos samples from the positive labels and ignore (-1) the remaining ones.
    # In some cases we get less than n_pos samples, in that we will randomly sample (n_sample - n_pos) negative samples (0)
    # and assign ignore label to the remaining anchor boxes. This is done using the following code
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1
    # negative index
    n_neg = n_sample - np.sum(label == 1)
    neg_index = np.where(label == 0)[0]

    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1
    return label


def corner_coordinates_format_to_center_format(boxes):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    ctr_y = boxes[:, 0] + 0.5 * height
    ctr_x = boxes[:, 1] + 0.5 * width
    return height, width, ctr_y, ctr_x


def center_format_to_corner_coordinates_format(ctr_x, ctr_y, h, w):
    boxes = np.zeros((len(ctr_x), 4), dtype=ctr_x.dtype)
    try:
        boxes[:, 0::4] = ctr_y - 0.5 * h
        boxes[:, 1::4] = ctr_x - 0.5 * w
        boxes[:, 2::4] = ctr_y + 0.5 * h
        boxes[:, 3::4] = ctr_x + 0.5 * w
    except RuntimeWarning:
        print(ctr_x, ctr_y, h, w)
    return boxes


def anchor_locs_to_center_format(anchor_locs, anc_height, anc_width, anc_ctr_y, anc_ctr_x):
    '''
    anchor_locs is a numpy data for anchor_locs predicted for a SINGLE image
    '''
    scale_clamp = np.log(1000.0 / 16)
    dy = anchor_locs[:, 0::4]
    dx = anchor_locs[:, 1::4]
    dh = anchor_locs[:, 2::4]
    dw = anchor_locs[:, 3::4]
    ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]

    dh = np.minimum(dh, scale_clamp)
    dw = np.minimum(dw, scale_clamp)

    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]

    return ctr_y, ctr_x, h, w


def find_box_differences(height, width, ctr_y, ctr_x, base_height, base_width, base_ctr_y, base_ctr_x):
    # Machine limits for floating point types: eps : float is the smallest representable positive number such that
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
    return anchor_locs


def get_anchor_locations(anchor_boxes, base_boxes):
    height, width, ctr_y, ctr_x = corner_coordinates_format_to_center_format(anchor_boxes)

    base_height, base_width, base_ctr_y, base_ctr_x = corner_coordinates_format_to_center_format(base_boxes)

    anchor_locs = find_box_differences(height, width, ctr_y, ctr_x, base_height, base_width, base_ctr_y, base_ctr_x)

    return anchor_locs


def get_locations_from_anchor_boxes(valid_anchor_boxes, bbox, argmax_ious):
    # find the ground-truth bounding box information for each valid anchor
    max_iou_bbox = bbox[argmax_ious]

    anchor_locs = get_anchor_locations(valid_anchor_boxes, max_iou_bbox)

    return anchor_locs


def process_anchors_locations_and_labels(anchors, index_inside, anchor_locs, label):
    """
    Returns
        anchor_locations [N, 4] - [22500, 4]
        anchor_labels [N, ] - [22500]
    """
    anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
    anchor_labels.fill(-1)  # by default all are ignored
    anchor_labels[index_inside] = label  # those valid are set to the sampled label (256 by now)
    anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
    anchor_locations.fill(0)  # set anchor locations for all as 0 (ignored)
    anchor_locations[index_inside, :] = anchor_locs  # those valid will have their locations
    return anchor_locations, anchor_labels


def generate_anchors_for_bbox(image_size, sub_sample, bbox, pos_iou_threshold, neg_iou_threshold, pos_ratio,
                              n_sample):
    # 1. Generate anchors and get their information
    anchors, valid_anchor_boxes, index_inside, argmax_ious, max_ious, gt_argmax_ious, gt_max_ious = get_ious_info(
        image_size, sub_sample, bbox)
    # 2. Generate the label information for the valid anchor boxes
    label = generate_labels_for_valid_anchors(valid_anchor_boxes, gt_argmax_ious, max_ious, pos_iou_threshold,
                                              neg_iou_threshold)
    # 3. Since there are still many, so we do a sampling processing here.
    sample_anchors(label, pos_ratio=pos_ratio, n_sample=n_sample)

    anchor_locs = get_locations_from_anchor_boxes(valid_anchor_boxes, bbox, argmax_ious)
    # 4. Process anchor locations (to the correct format) and labels (-1, 0, 1)
    anchor_locations, anchor_labels = process_anchors_locations_and_labels(anchors, index_inside, anchor_locs, label)
    return anchors, anchor_locations, anchor_labels
