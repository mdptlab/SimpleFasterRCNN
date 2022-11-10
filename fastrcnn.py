import numpy as np
import torch
import torch.nn as nn


def prepare_data_for_fast_rcnn(sample_roi, image_idx):
    rois = torch.from_numpy(sample_roi).float()
    roi_indices = image_idx * np.ones((len(rois),), dtype=np.int32)  # TODO: This should be changed to the image id
    roi_indices = torch.from_numpy(roi_indices).float()
    indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
    # [image_id, x1, y1, x2, y2] format
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    return indices_and_rois, xy_indices_and_rois


def adaptive_pooling(out_map, sample_roi, image_idx, sub_sample):
    indices_and_rois, xy_indices_and_rois = prepare_data_for_fast_rcnn(sample_roi, image_idx)
    size = (7, 7)
    adaptive_max_pool = nn.AdaptiveMaxPool2d(size)
    output = []
    rois = indices_and_rois.data.float()
    # subsampling ratio (get from the image ratio to feature ratio)
    rois[:, 1:].mul_(1 / float(sub_sample))
    rois = rois.long()
    num_rois = rois.size(0)

    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]

        output.append(adaptive_max_pool(im))

    if len(output) == 0:
        print(output)
    output = torch.cat(output, 0)
    # Reshape the tensor so we can pass it through the feed forward layer
    k = output.view(output.size(0), -1)
    return k


class FastRCNN(nn.Module):
    def __init__(self, num_of_classes):
        super(FastRCNN, self).__init__()
        num_of_classes_and_bg = num_of_classes + 1
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)])
        self.cls_loc = nn.Linear(4096, num_of_classes_and_bg * 4)  # each will have 4 coordinates per class
        self.cls_loc.weight.data.normal_(0, 0.01)  # TODO: Should check for different methods of normalization
        self.cls_loc.bias.data.zero_()
        self.score = nn.Linear(4096, num_of_classes_and_bg)
        self.score.weight.data.normal_(0, 0.01)
        self.score.bias.data.zero_()

    def forward(self, k):
        # forward
        # passing the output of roi-pooling to the network defined above we get
        k = self.roi_head_classifier(k)
        roi_cls_loc = self.cls_loc(k)
        roi_cls_score = self.score(k)
        return roi_cls_loc, roi_cls_score
