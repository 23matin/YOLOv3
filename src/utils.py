from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import cv2
import random
import pickle as pkl
from modules import *

def get_torch_version():
    return torch.__version__

def prase_cfg(cfg_file):
    """
    Takes in a configfile and return a list of dicts.
    """
    print("=============================================================================")
    print("[INFO] start to prase config file {}.".format(cfg_file))

    all_type = set()
    with open(cfg_file) as fin:
        lines = fin.readlines()
        lines = [x.lstrip().rstrip()
                 for x in lines if len(x.lstrip().rstrip()) > 0 and x[0] != '#']
        blocks = []
        block = {}
        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                btype = line[1:-1].rstrip()
                block["btype"] = btype
                all_type.add(btype)
            else:
                key, val = line.split("=")
                block[key.lstrip().rstrip()] = val.rstrip().lstrip()
        blocks.append(block)

    print("Summary:")
    print("Net have {} blocks with {} types:{}".format(
        len(blocks), len(all_type), all_type))
    print("============================================================================")

    return blocks

def str2int_or_float(string):
    if not isinstance(string, str):
        raise Exception("Not a valid string.")
    try:
        return int(string)
    except Exception as e:
        try:
            return float(string)
        except Exception as e:
            raise Exception(
                "Falied to convert '{}' to a number.".format(string))

def create_modules(block_cfg):
    prev_filter = 3
    output_filters = []
    modules = nn.ModuleList()
    for idx, block in enumerate(block_cfg[1:]):
        btype = block["btype"]
        if btype == "convolutional":
            # get params
            inp = prev_filter
            oup = int(block["filters"])
            bn = "batch_normalize" in block
            bias = not bn  # 如果没有BN，conv要加上bias
            stride = int(block["stride"])
            kernel = int(block["size"])
            actv_fn = block["activation"]
            prev_filter = oup
            module = make_conv_bn(inp, oup, bn, actv_fn, kernel, stride, int(block["pad"]), idx=idx)
        elif btype in ["shortcut"]:
            if idx > 0:
                module = nn.Sequential(OrderedDict(
                    [("{}_{}".format(btype, idx), EmptyLayer(btype, block))]))
            else:
                module = nn.Sequential(EmptyLayer(btype, block))
        elif btype in ["route"]:
            if idx > 0:
                module = nn.Sequential(OrderedDict(
                    [("{}_{}".format(btype, idx), EmptyLayer(btype, block))]))
            else:
                module = nn.Sequential(EmptyLayer(btype, block))
            idxs = [str2int_or_float(s) for s in block["layers"].split(",")]
            filters = [output_filters[i] for i in idxs]
            prev_filter = sum(filters)
        elif btype == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            num_classes = int(block["classes"])
            img_size = int(block["size"])
            if idx > 0:
                module = nn.Sequential(OrderedDict(
                    [("yolo_{}".format(idx), YOLODetLayer(anchors, num_classes, img_size))]))
            else:
                module = nn.Sequential(YOLODetLayer(anchors, num_classes, img_size))
        elif btype == "upsample":
            seq = nn.Sequential()
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            if idx > 0:
                module = nn.Sequential(OrderedDict(
                    [("upsample_{}".format(idx), upsample)]))
            else:
                module = nn.Sequential(upsample)
        else:
            raise Exception(
                "Block type {} not support for now.".format(btype))
        modules.append(module)
        output_filters.append(prev_filter)
    return modules

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes

    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


# 类似于 set
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask  # 将置信度小于阈值的直接置为0

    # convert to corner
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    output = None

    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor

        # get max conf and max_conf_score and append to predict
        max_conf, max_conf_score = torch.max(
            image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        # obj confidence > threshold
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)  # (N,7) N:object confidience 大于 confidence 的框的个数,7 :x0,y0,x1,y1,obj_conf,max_conf(对应class),max_conf_score
        except:
            continue

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue

        for cls in unique(image_pred_[:,-1]):
            # TODO:

            # get the detections with one particular class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                        0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
                # image_pred_class: 每一类经过NMS后剩余的 N*7的向量

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
            ind)  # Repeat the batch_id for as many detections of
            # the class cls in the image
            class_out = torch.cat((batch_ind, image_pred_class), 1)
            if output is None:
                output = class_out
            else:
                output = torch.cat((output, class_out), 0)
    return output


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) //
           2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def letterbox_convert_back(img, inp_dim, x, y):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    resize_rate = min(w / img_w, h / img_h)
    # print("resize rate: ",resize_rate)
    x = img_w/2+(x - w / 2) / resize_rate
    y = img_h/2+(y - h / 2) / resize_rate
    return x, y
