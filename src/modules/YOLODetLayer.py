import numpy as np
import torch
import torch.nn as nn


class YOLODetLayer(nn.Module):
    """
    在YOLOv3的理解中，没有neck的概念，yolov3 的cfg中的YOLO网络只有将detection结果decode的部分。
    """

    def __init__(self, anchors, num_classes, img_size):
        super(YOLODetLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x, use_cuda=False):
        return decode_yolov3(x, self.anchors, self.num_classes, self.img_size, use_cuda)


def decode_yolov3(x, anchors, num_classes, img_size, use_cuda=False):
    """
    "num_classes"
    "anchors"
    "img_size"
    "use_cuda"
    """
    batch_size = x.size(0)
    stride = img_size // x.size(-1)
    num_per_box_pred = 5 + num_classes
    anchors = anchors
    grid_size = x.size(-1)  # ??
    grid_size = img_size // stride

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

    if use_cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        anchors = torch.FloatTensor(anchors)
        anchors = anchors.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, len(anchors)).view(-1, 2).unsqueeze(0)

    x[:, :, :2] += x_y_offset

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * anchors
    x[:, :, :2] *= stride
    return x
