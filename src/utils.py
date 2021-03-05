from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict


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
                block[key.lstrip()] = val.rstrip()
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


class BlockCreater(object):
    """
        create blocks of different type and return an obdect of type nn.Sequence()
        supported types are: 'route', 'convolutional', 'upsample', 'shortcut', 'yolo'
        and will name the module accroding idx if idx is support
    """

    def __init__(self):
        self.init_channels = 3
        self.inp = self.init_channels

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

    def __create_yolo(block, idx=-1):
        mask = block["mask"].split(",")
        mask = [int(x) for x in mask]
        anchors = x["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i+1])
                   for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in mask]

        print(mask)
        print(anchors)

    def create(self, block, idx=-1):
        btype = block["btype"]
        if btype == "convolutional":
            return self.__create_conv(block, idx=idx)
        elif btype == "upsample":
            seq = nn.Sequential()
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            if idx > 0:
                seq.add_module("upsample_{}".format(idx), upsample)
            else:
                seq.add_module(upsample)
            return seq
        elif btype in ["shortcut", "route"]:
            if idx > 0:
                return nn.Sequential(OrderedDict(
                    [("route_{}".format(idx), EmptyLayer("route", block))]))
            else:
                nn.Sequential(EmptyLayer("route", block))
        elif btype == "yolo":
            self.__create_yolo(block, idx=idx)
        else:
            raise Exception(
                "Block type {} not support for now.".format(btype))

def ensemble_model():
    pass
