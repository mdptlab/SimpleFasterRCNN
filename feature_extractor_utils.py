# Backbone
import torchvision
import torch
import torch.nn as nn
import functools


@functools.lru_cache(maxsize=1)
def get_backbone(image_size=800, sub_sample=16):
    dummy_img = torch.zeros((1, 3, image_size, image_size)).float()
    model = torchvision.models.vgg16(pretrained=True)
    fe = list(model.features)
    req_features = []
    k = dummy_img.clone()
    for i in fe:
        k = i(k)
        if k.size()[2] < 800//sub_sample:
            break
        req_features.append(i)
        out_channels = k.size()[1]
    faster_rcnn_fe_extractor = nn.Sequential(*req_features)
    return faster_rcnn_fe_extractor


def extract_base_feature(image, image_size=800, sub_sample=16):
    faster_rcnn_fe_extractor = get_backbone(image_size, sub_sample)
    faster_rcnn_fe_extractor.eval()
    out_map = faster_rcnn_fe_extractor(image)
    return out_map
