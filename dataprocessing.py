import os
from xml.etree import ElementTree

import cv2
import numpy as np


# exporting data
def get_data(country, damage_id_mappings, base_path):
    image_path = base_path + '/train/' + country + '/images'
    annotation_path = base_path + '/train/' + country + '/annotations/xmls/'
    file_list = [filename.split('.')[0] for filename in os.listdir(image_path) if not filename.startswith('.')]

    images = []
    bounding_boxes = []
    labels = []
    image_ratio = []
    for file_name in file_list:
        if file_name == '.DS_Store':
            pass
        else:
            # the image
            img = cv2.imread(image_path + '/' + file_name + '.jpg')
            this_image_ratio = img.shape[0] / 800
            image_ratio.append(this_image_ratio)
            img = cv2.resize(img, (800, 800))
            # the labels and bounding boxes
            infile_xml = open(annotation_path + '/' + file_name + '.xml')
            tree = ElementTree.parse(infile_xml)
            root = tree.getroot()
            this_image_labels = []
            this_image_bboxes = []

            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name in damage_id_mappings.keys():
                    # labels
                    this_image_labels.append(damage_id_mappings[cls_name])
                    # bounding box
                    xmlbox = obj.find('bndbox')
                    xmin = int(xmlbox.find('xmin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymin = int(xmlbox.find('ymin').text)
                    ymax = int(xmlbox.find('ymax').text)
                    # the format is y1, x1, y2, x2
                    this_image_bboxes.append(((np.array([ymin, xmin, ymax, xmax]) / this_image_ratio)).astype(int))
            if len(this_image_labels) > 0:
                images.append(img)
                labels.append(np.array(this_image_labels).astype(int))
                bounding_boxes.append(np.array(this_image_bboxes))
    return images, labels, bounding_boxes
