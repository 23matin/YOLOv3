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


class EmptyLayer(nn.Module):
    """
    mainly use to form shortcut or rotute layers
    """

    def __init__(self, layer_type, data=None):
        super(EmptyLayer, self).__init__()

        self.type = layer_type
        self.data = data


class YOLODetLayer(nn.Module):
    """
    在YOLOv3的理解中，没有neck的概念，yolov3 的cfg中的YOLO网络只有将detection结果decode的部分。
    """

    def __init__(self, anchors):
        super(YOLODetLayer, self).__init__()
        self.anchors = anchors


class BlockCreater(object):
    """
        create blocks of different type and return an obdect of type nn.Sequence()
        supported types are: 'route', 'convolutional', 'upsample', 'shortcut', 'yolo'
        and will name the module accroding idx if idx is supported
    """

    def __init__(self):
        self.init_channels = 3
        self.inp = self.init_channels
        self.output_filters = [self.init_channels]

    def __create_conv(self, block, idx=-1):
        # get params
        if "prev_filters" in block:
            self.inp = block["prev_filters"]
        oup = int(block["filters"])
        bn = "batch_normalize" in block
        bias = not bn  # 如果没有BN，conv要加上bias
        stride = int(block["stride"])
        kernel = int(block["size"])
        if int(block["pad"]):
            pad = kernel // 2
        else:
            pad = 0
        actv_fn = block["activation"]

        # convert to nn.Module
        seq = nn.Sequential()
        conv_module = nn.Conv2d(
            self.inp, oup, kernel, stride, pad, bias=bias)
        bn_module = nn.BatchNorm2d(oup) if bn else None
        activation_module = None
        if actv_fn == "leaky":
            activation_module = nn.LeakyReLU(0.1, inplace=True)
        elif actv_fn == "relu":
            activation_module = nn.ReLU(inplace=True)
        elif actv_fn == "linear":
            pass
        else:
            raise Exception(
                "activation type '{}' is not support for now!".format(actv_fn))

        self.inp = oup  # update num_filters
        self.output_filters.append(oup)

        # ensemble
        if idx > 0:
            seq.add_module("conv_{}".format(idx), conv_module)
            if bn_module:
                seq.add_module("bn_{}".format(idx), bn_module)
            if activation_module:
                seq.add_module("{}_{}".format(
                    actv_fn, idx), activation_module)
        else:
            seq.add_module(conv_module)
            if bn_module:
                seq.add_module(bn_module)
            if activation_module:
                seq.add_module(activation_module)
        return seq

    def __create_yolo(self, block, idx=-1):
        mask = block["mask"].split(",")
        mask = [int(x) for x in mask]
        anchors = block["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i + 1])
                   for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in mask]

        # print(mask)
        # print(anchors)
        if idx > 0:
            return nn.Sequential(OrderedDict(
                [("yolo_{}".format(idx), YOLODetLayer(anchors))]))
        else:
            return nn.Sequential(YOLODetLayer(anchors))

    def create(self, block, idx=-1):
        btype = block["btype"]
        if btype == "convolutional":
            module = self.__create_conv(block, idx=idx)
        elif btype == "upsample":
            seq = nn.Sequential()
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride,
                                   mode="nearest", align_corners=False)
            if idx > 0:
                module = nn.Sequential(OrderedDict(
                    [("upsample_{}".format(idx), upsample)]))
            else:
                module = nn.Sequential(upsample)
            self.output_filters.append(self.output_filters[-1])
        elif btype in ["shortcut"]:
            if idx > 0:
                module = nn.Sequential(OrderedDict(
                    [("{}_{}".format(btype, idx), EmptyLayer(btype, block))]))
            else:
                module = nn.Sequential(EmptyLayer(btype, block))
            self.output_filters.append(self.output_filters[-1])
        elif btype in ["route"]:
            if idx > 0:
                module = nn.Sequential(OrderedDict(
                    [("{}_{}".format(btype, idx), EmptyLayer(btype, block))]))
            else:
                module = nn.Sequential(EmptyLayer(btype, block))
            idxs = [str2int_or_float(s) for s in block["layers"].split(",")]
            idxs = [x + idx if x < 0 else x for x in idxs]
            num_filters = [self.output_filters[i - 1] for i in idxs]
            self.output_filters.append(sum(num_filters))
        elif btype == "yolo":
            module = self.__create_yolo(block, idx=idx)
            self.output_filters.append(self.output_filters[-1])
        else:
            raise Exception(
                "Block type {} not support for now.".format(btype))

        return module, btype


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


def decode_yolov3(x, cfg):
    """
    cfg should contain:
    "num_classes"
    "anchors"
    "img_size"
    "use_cuda"
    """
    batch_size = x.size(0)
    stride = cfg["img_size"] // x.size(-1)
    num_per_box_pred = 5 + cfg["num_classes"]
    anchors = cfg["anchors"]
    grid_size = x.size(-1)  # ??
    grid_size = cfg["img_size"] // stride

    # merge all predictions
    x = x.view(batch_size, num_per_box_pred *
               len(anchors), grid_size * grid_size)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, grid_size * grid_size *
               len(anchors), num_per_box_pred)
    # anchors = [(a[0] / stride, a[1] / stride) for a in anchors]  # ??

    x[:, :, 0] = torch.sigmoid(x[:, :, 0])
    x[:, :, 1] = torch.sigmoid(x[:, :, 1])
    x[:, :, 4:] = torch.sigmoid(x[:, :, 4:])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if cfg["use_cuda"]:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, len(anchors)).view(-1, 2).unsqueeze(0)

    x[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if cfg["use_cuda"]:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * anchors
    x[:, :, :2] *= stride
    return x


def get_decode_fn(name):
    """
    decode function factory including:
    yolov3
    """
    decoder_dict = {"yolov3": decode_yolov3}
    if name not in decoder_dict:
        raise Exception(
            "ObjectDetectionDecoder type {} not supported.".format(name))
    return decoder_dict[name]


class ObjectDetectionDecoder(object):
    def __init__(self, name, cfg=None):
        self.cfg = cfg
        self.name = name
        self.decode_fn = get_decode_fn(self.name)

    def decode(self, x):
        return self.decode_fn(x, self.cfg)

    def set_cfg(self, cfg):
        self.cfg = cfg


class NetWorkByCfg(nn.Module):
    """
    This class is a super class which use cfg file to define a network.
    this class will parse config file and generate:
    1.  self.module_list
    2.  self.net_info
    """

    def __init__(self, cfg_file):
        super(NetWorkByCfg, self).__init__()

        # 1. prase config file
        blocks = prase_cfg(cfg_file)

        # 2. Store all params hyperparameters in net_info if there is any
        self.net_info = {}
        for key, val in blocks[0].items():
            self.net_info[key] = val

        # 3. Use BlockCreater to get all blocks of type (nn.ModuleList) according to a .cfg file
        self.module_list = nn.ModuleList()
        self.module_type_list = list()
        bc = BlockCreater()
        for idx, block in enumerate(blocks):
            if block["btype"] == "net":
                continue
            module, module_type = bc.create(block, idx)
            self.module_list.append(module)
            self.module_type_list.append(module_type)


# class Darknet(NetWorkByCfg):
#     def __init__(self, cfg_file, use_cuda=False):
#         super(Darknet, self).__init__(cfg_file)
#         self.use_cuda = False
#         self.decode_cfg = dict()
#         self.decode_cfg["img_size"] = int(self.net_info["width"])
#         self.decode_cfg["num_classes"] = int(self.net_info["classes"])
#         self.decode_cfg["use_cuda"] = use_cuda
#         self.decode_cfg["anchors"] = None  # dynamic set
#         self.decoder = ObjectDetectionDecoder("yolov3", self.decode_cfg)
#
#     def forward(self, x):
#         outputs = dict()
#         output_filters = []
#         output_filters.append(x.size(1))
#         detections = None  # accumulate decoded detections
#         for idx, module_info in enumerate(zip(self.module_list, self.module_type_list)):
#             module, module_type = module_info
#             if module_type in ["convolutional", "upsample"]:
#                 print("==============idx:", idx)
#                 x = module(x)
#             elif module_type == "shortcut":
#                 data = module[0].data
#                 data_from = int(data["from"])
#                 x = outputs[idx - 1] + outputs[idx + data_from]
#             elif module_type == "route":
#                 merge_layers = module[0].data["layers"].split(",")
#                 merge_layers = [int(l) for l in merge_layers]
#                 if len(merge_layers) == 1:
#                     x = outputs[idx + merge_layers[0]]
#                 else:
#                     concate_list = list()
#                     for route_idx in merge_layers:
#                         if route_idx > 0:
#                             route_idx -= (idx - 1)  # 因为将开始存储超参数的net类型已经去掉了
#                         concate_list.append(outputs[route_idx])
#                     x = torch.cat(concate_list, dim=1)
#             elif module_type == "yolo":
#                 self.decode_cfg["anchors"] = module[0].anchors  # change anchors when anchors change
#                 self.decoder.set_cfg(self.decode_cfg)
#                 det = self.decoder.decode(x)
#                 if detections is None:
#                     detections = det
#                 else:
#                     detections = torch.cat((detections, det), 1)
#             outputs[idx] = x
#         return detections

def create_yolo(block, idx=-1):
    mask = block["mask"].split(",")
    mask = [int(x) for x in mask]
    anchors = block["anchors"].split(",")
    anchors = [int(a) for a in anchors]
    anchors = [(anchors[i], anchors[i + 1])
               for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    # print(mask)
    # print(anchors)
    if idx > 0:
        return nn.Sequential(OrderedDict(
            [("yolo_{}".format(idx), YOLODetLayer(anchors))]))
    else:
        return nn.Sequential(YOLODetLayer(anchors))


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
            if int(block["pad"]):
                pad = kernel // 2
            else:
                pad = 0
            actv_fn = block["activation"]

            # convert to nn.Module
            module = nn.Sequential()
            conv_module = nn.Conv2d(
                inp, oup, kernel, stride, pad, bias=bias)
            bn_module = nn.BatchNorm2d(oup) if bn else None
            activation_module = None
            if actv_fn == "leaky":
                activation_module = nn.LeakyReLU(0.1, inplace=True)
            elif actv_fn == "relu":
                activation_module = nn.ReLU(inplace=True)
            elif actv_fn == "linear":
                pass
            else:
                raise Exception(
                    "activation type '{}' is not support for now!".format(actv_fn))

            prev_filter = oup  # update num_filters

            # ensemble
            if idx >= 0:
                module.add_module("conv_{}".format(idx), conv_module)
                if bn_module:
                    module.add_module("bn_{}".format(idx), bn_module)
                if activation_module:
                    module.add_module("{}_{}".format(
                        actv_fn, idx), activation_module)
            else:
                module.add_module(conv_module)
                if bn_module:
                    module.add_module(bn_module)
                if activation_module:
                    module.add_module(activation_module)
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
            module = create_yolo(block, idx=idx)
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


class Darknet(nn.Module):
    def __init__(self, cfg_file, use_cuda=False):
        super(Darknet, self).__init__()
        self.use_cuda = False
        self.block_cfg = prase_cfg(cfg_file)
        self.module_list = create_modules(self.block_cfg)
        self.module_type_list = [x["btype"]
                                 for x in self.block_cfg if x["btype"] != "net"]

        self.net_info = {}
        for key, val in self.block_cfg[0].items():
            self.net_info[key] = val

        self.decode_cfg = dict()
        self.decode_cfg["img_size"] = int(self.net_info["width"])
        self.decode_cfg["num_classes"] = int(self.net_info["classes"])
        self.decode_cfg["use_cuda"] = use_cuda
        self.decode_cfg["anchors"] = None  # dynamic set
        self.decoder = ObjectDetectionDecoder("yolov3", self.decode_cfg)

    def forward(self, x):
        outputs = dict()
        output_filters = []
        output_filters.append(x.size(1))
        detections = None  # accumulate decoded detections
        for idx, module_info in enumerate(zip(self.module_list, self.module_type_list)):
            module, module_type = module_info
            if module_type in ["convolutional", "upsample"]:
                x = module(x)
            elif module_type == "shortcut":
                data = module[0].data
                data_from = int(data["from"])
                x = outputs[idx - 1] + outputs[idx + data_from]
            elif module_type == "route":
                merge_layers = module[0].data["layers"].split(",")
                merge_layers = [int(l) for l in merge_layers]
                if len(merge_layers) == 1:
                    x = outputs[idx + merge_layers[0]]
                    # print("merge: ",idx + merge_layers[0])
                else:
                    concate_list = list()
                    # print("merge_layers: ", merge_layers)
                    for route_idx in merge_layers:
                        if route_idx > 0:
                            route_idx -= idx
                        # print("merge: ",idx + route_idx)
                        concate_list.append(outputs[idx + route_idx])
                    x = torch.cat(concate_list, dim=1)
                    # print("merged x:",x)
            elif module_type == "yolo":
                # change anchors when anchors change
                self.decode_cfg["anchors"] = module[0].anchors
                self.decoder.set_cfg(self.decode_cfg)
                x = self.decoder.decode(x)
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[idx] = x
            if False:
                print("type: ", module_type)
                print("outputs[{}]".format(idx), outputs[idx])
        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.block_cfg[i + 1]["btype"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(
                        self.block_cfg[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                # If module_type is convolutional load weights
                # Otherwise ignore.
                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


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
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1,
                                                                     7)  # (N,7) N:object confidience 大于 confidence 的框的个数,7 :x0,y0,x1,y1,obj_conf,max_conf(对应class),max_conf_score
        except:
            continue

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue
        print("img_classes: ", image_pred_)

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
    print("resize rate: ",resize_rate)
    x = img_w/2+(x - w / 2) / resize_rate
    y = img_h/2+(y - h / 2) / resize_rate
    return x, y
