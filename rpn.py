import torch.nn as nn

class RPN(nn.Module):
    def __init__(self, mid_channels=512, in_channels=512, n_anchor=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, kernel_size=1, stride=1, padding=0) # * 4 for four locs

        # We will use softmax here and so, we put n_classes
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, kernel_size=1, stride=1, padding=0) # *2 for 0 or 1 cls
        # conv sliding layer
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()

        # Regression layer
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        # Classification layer
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def forward(self, out_map):
        x = self.conv1(out_map)  # out_map is obtained in section 1
        pred_anchor_locs = self.reg_layer(x)
        pred_cls_scores = self.cls_layer(x)

        # Reformat the structure so we can see that for each image, each of the feature map (50x50)
        # - there are 18 class predictions (2 for each of the 9 anchors extracted for each feature pixel)
        # - there are 36 location values (4 for each of the 9 anchors extracted fro each feature pixel)
        # so there are 22500 anchors => so we have these 22500 locations, each has 4 values
        # TODO: 1 here in the first dimension is the number of images (1 in this case)
        pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

        # predicted class scores
        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()

        # [:, :, :, :, 1], the 1 here means the 2nd bounding box, so the objectness score is for second one?
        rpn_softmax_scores = nn.functional.softmax(pred_cls_scores.view(1, 50, 50, 9, 2), dim=4)
        objectness_score = rpn_softmax_scores[:, :, :, :, 1].contiguous().view(1, -1)

        # Convert it to pred cls scores => 2 here means two classes, one for foreground (1) another for background (0)
        pred_cls_scores = pred_cls_scores.view(1, -1, 2)

        return pred_anchor_locs, objectness_score, pred_cls_scores
