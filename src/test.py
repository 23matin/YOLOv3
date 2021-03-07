from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *


def test_cfg():
    cfg_file = "/home/matin23/workspace/YOLOv3/cfg/yolov3.cfg"
    print(prase_cfg(cfg_file))


def testBlockCreater():
    bc = BlockCreater()
    cfg_file = "../cfg/yolov3.cfg"
    cfg = prase_cfg(cfg_file)
    test_target = ["convolutional", "upsample", "shortcut", "route", "yolo"]
    for item in cfg:
        if len(test_target) == 0:
            return
        if item["btype"] in test_target:
            block, module_type = bc.create(item, 1)
            print("[{} example:]".format(item["btype"]))
            print(block)
            print(block.type)
            test_target.remove(item["btype"])
            print()


def test_list():
    a = [1, 2, 3]
    b = [4, 5, 6]
    for idx, item in enumerate(zip(a, b)):
        print(idx)
        print(item)


def test_str2int_or_float():
    case = ['0', '0.5', 'e5']
    for item in case:
        res = str2int_or_float(item)
        print("str: {}  number: {}  type:{}".format(item, res, type(res)))


def test_dict():
    import collections
    print('\nOrderedDict:')
    d1 = collections.OrderedDict()
    d1['a'] = 'A'
    d1['b'] = 'B'
    d1['c'] = 'C'

    d2 = collections.OrderedDict()
    d2['c'] = 'C'
    d2['a'] = 'A'
    d2['b'] = 'B'

    # for item in d1:
    #     print(d1)

    for item in d2:
        print(d1)
    # print(d1)
    # print(d2)


def test_simple():
    # test_cfg()
    # testBlockCreater()
    # test_dict()
    # test_str2int_or_float()
    test_list()


test_simple()
