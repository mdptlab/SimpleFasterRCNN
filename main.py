import numpy as np
import torch
import torch.optim as optim

from faster_rcnn import FasterRCNN
from feature_extractor_utils import get_backbone
from dataprocessing import get_data
from faster_rcnn_losses import calculate_faster_rcnn_loss
from trainer import training_loop


if __name__ == '__main__':
    # test data
    bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32)  # [y1, x1, y2, x2] format
    labels = np.array([1, 2], dtype=np.int8)  # 0 represents background
    image_size = 800
    sub_sample = 16

    damage_types = ["D00", "D10", "D20", "D40"]
    damage_names = ['Longitudinal Crack', 'Transverse Crack', 'Aligator Crack', 'Pothole']
    type_name_mappings = {'D00': 'Longitudinal Crack', 'D10': 'Transverse Crack', 'D20': 'Aligator Crack',
                          'D40': 'Pothole'}
    damage_id_mappings = {'D00': 1, 'D10': 2, 'D20': 3, 'D40': 4}
    damage_ids = [1, 2, 3, 4]

    # Use CUDA or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    image = torch.zeros((1, 3, 800, 800)).float().to(device)

    # Settings for Regional Proposal Network
    mid_channels = 512
    in_channels = 512  # depends on the output feature map in vgg 16 it is 512
    n_anchor = 9  # Number of anchors at each location
    n_classes = 4

    faster_rcnn_fe_extractor = get_backbone().to(device)
    model = FasterRCNN(faster_rcnn_fe_extractor, image_size, sub_sample, mid_channels=mid_channels,
                       in_channels=in_channels, n_anchor=n_anchor, n_classes=n_classes, device=torch.device("cpu")).to(
        device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

    images, labels, bounding_boxes = get_data('Czech', damage_id_mappings, '../RoadDamageDetector')

    # train
    train_data = zip(images, labels, bounding_boxes)
    loss_fn = calculate_faster_rcnn_loss
    val_data = None
    n_epochs = 2
    training_loop(n_epochs, optimizer, model, device, loss_fn, train_data, val_data)
