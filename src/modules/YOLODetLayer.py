import numpy as np
import torch
import torch.nn as nn


def bbox_wh_iou(wh1, wh2):
    """
    忽略中心的偏移
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


class YOLODetLayer(nn.Module):
    """
    在YOLOv3的理解中，没有neck的概念，yolov3 的cfg中的YOLO网络只有将detection结果decode的部分。
    """

    def __init__(self, anchors, num_classes, img_size):
        super(YOLODetLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.ignore_thredshold = 0.5
        # !!attention: anchor [11,11] 是对应的416*416 size 的11 也就是说在13*13的特征图上的归一化的xw应该是 11/416*13
        self.scaled_anchors = None
        self.grid_size = 0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.obj_scale = 1
        self.noobj_scale = 100
    

    def build_mask(self, dets, target):
        """
        target of format:
        [batch_id, class_id，cx, cy, w, h]

        dets of format:
        (batch,nums_class+5,grid_size,grid_size,num_anchors)
        """
        n_batch, n_anchors, n_grid = dets.size(0), len(self.anchors), self.grid_size

        BoolTensor = torch.cuda.BoolTensor if dets.is_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if dets.is_cuda else torch.FloatTensor

        anchors = FloatTensor(self.scaled_anchors)

        obj_mask = BoolTensor(n_batch, n_grid, n_grid, n_anchors).fill_(0)
        noobj_mask = BoolTensor(n_batch, n_grid, n_grid, n_anchors).fill_(1)
        t_obj_conf = FloatTensor(n_batch, n_grid, n_grid, n_anchors).fill_(0)
        tx = FloatTensor(n_batch, n_grid, n_grid, n_anchors).fill_(0)
        ty = FloatTensor(n_batch, n_grid, n_grid, n_anchors).fill_(0)
        th = FloatTensor(n_batch, n_grid, n_grid, n_anchors).fill_(0)
        tw = FloatTensor(n_batch, n_grid, n_grid, n_anchors).fill_(0)
        tcls = FloatTensor(n_batch, n_grid,
                          n_grid, n_anchors, self.num_classes).fill_(0)

        # in pixel coordianate
        # !!attention: 所有gtbox在pixel坐标系下的坐标  of shale [n_gt,4]
        target_boxes = target[:, 2:6] * n_grid
        gwh = target_boxes[:, 2:]  # shape [n,2]
        ious = torch.stack([bbox_wh_iou(anchor, gwh)
                           for anchor in anchors])  # of shape [n_gt,n_anchors]
        best_iou, anchor_id = ious.max(dim=0)  # (n_gt)

        gxy = target_boxes[:, :2]
        gi, gj = gxy.long().t()

        # find the best iou and assign obj_mask
        batch_idx = target[:, 0].long().t()  # !!转换为整数然后转置
        # !!attention: batch_idx, anchor_id, gxy[0].t(), gxy[1].t() should be in same size and is a row vector
        obj_mask[batch_idx,  gj, gi,anchor_id] = True

        noobj_mask[batch_idx, gj, gi,anchor_id] = False

        # iou > thredshold
        # !!attention: 非负和非正的样本只存在于GT对应的那个格子的非最好的其他的anchors，其他格子的都是负样本!!! 如果最好iou的都没有达到ignore_thres 那么这个目标GT没有正样本框
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[batch_idx[i], gj[i], gi[i], anchor_ious >self.ignore_thredshold] = False

        # cx cy w h
        offset = gxy - gxy.floor()
        tx[batch_idx, gj, gi, anchor_id] = offset[:,0]
        ty[batch_idx, gj, gi, anchor_id] = offset[:,1]

        tw[batch_idx, gj, gi, anchor_id] = torch.log(
            gwh[:, 0] / anchors[anchor_id][:, 0] + 1e-16)
        th[batch_idx, gj, gi, anchor_id] = torch.log(
            gwh[:, 1] / anchors[anchor_id][:, 1] + 1e-16)

        target_lables = target[:, 1].long().t()
        tcls[batch_idx, gj, gi, anchor_id, target_lables] = True

        t_obj_conf[obj_mask] = 1.0

        return obj_mask, noobj_mask, tcls, tx, ty, tw, th, t_obj_conf, tcls

    def forward(self, x, target=None, use_cuda=False):
        """
        target of format:
        [batch_id, class_id，cx, cy, w, h]
        """
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        # if target is None:
        # if False:
        if False:
            return decode_yolov3(x, self.anchors, self.num_classes, self.img_size, use_cuda), 0
        else:
            # original x is of shape: [batch_id, len(anchors) * (self.num_classes+5), grid_size,grid_size]
            if x.size(2) != self.grid_size:
                self.grid_size = x.size(2)
                self.scaled_anchors = FloatTensor([[anchor[0]/self.img_size*self.grid_size, anchor[1]/self.img_size*self.grid_size]
                                       for anchor in self.anchors])
                self.grid_coord_x = torch.arange(self.grid_size).repeat(
                    self.grid_size, 1).view([1, self.grid_size, self.grid_size , 1]).type(FloatTensor)
                self.grid_coord_y = torch.arange(self.grid_size).repeat(
                    self.grid_size, 1).t().view([1,  self.grid_size, self.grid_size,1]).type(FloatTensor)
                self.grid_w_in_pixel = FloatTensor(self.scaled_anchors[:, 0].view(
                    1,  1, 1,self.scaled_anchors.size(0))).type(FloatTensor)
                self.grid_h_in_pixel = FloatTensor(self.scaled_anchors[:, 1].view(
                    1, 1, 1,self.scaled_anchors.size(0))).type(FloatTensor)
                self.stride = self.img_size / self.grid_size

            n_batch = x.size(0)
            prediction = (
                x.view(n_batch, len(self.anchors),
                    self.num_classes + 5, self.grid_size, self.grid_size)
                # (batch,grid_size,grid_size,num_anchors,self.num_classes + 5)
                .permute(0, 3, 4, 1, 2)
                .contiguous()

            )
            # x = x.view(batch_size, num_per_box_pred *
            #    len(anchors), grid_size * grid_size)
            # x = x.transpose(1, 2).contiguous()
            # x = x.view(batch_size, grid_size * grid_size *
            #         len(anchors), num_per_box_pred)
            # self.num_classes + 5: x,y,w,h obj_p class_p

            # decode
            pred_obj_conf = torch.sigmoid(prediction[..., 4])
            pred_class = torch.sigmoid(prediction[..., 5:])
            # (batch,nums_class+5,grid_size,grid_size,4）
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            # pred_boxes[..., :2] *= self.grid_size  # prediction in grid pixel size

             # !!attention: used to calculate loss
            x, y, w, h = torch.sigmoid(prediction[..., 0]), torch.sigmoid(prediction[...,1]), prediction[..., 2], prediction[..., 3]


            # decode
            pred_boxes[..., :2] = torch.sigmoid(prediction[..., :2])
            pred_boxes[..., 0] = pred_boxes[..., 0] + self.grid_coord_x  # !!attention: broad cast
            pred_boxes[..., 1] = pred_boxes[..., 1] + self.grid_coord_y
            pred_boxes[..., 2:4] = torch.exp(prediction[..., 2:4])
            pred_boxes[..., 2] *= self.grid_w_in_pixel
            pred_boxes[..., 3] *= self.grid_h_in_pixel

            outputs = torch.cat(
                (
                    # !!attention: 这样一来的x,y,w,h,都是 原图size的 416 416 下的像素坐标！！！
                    pred_boxes.view(n_batch, -1, 4) * self.stride,
                    pred_obj_conf.view(n_batch, -1, 1),
                    pred_class.view(n_batch, -1, self.num_classes)
                ),
                -1
            )
            if target is None:
                return outputs, 0.

            obj_mask, noobj_mask, tcls, tx, ty, tw, th, t_obj_conf, tcls = self.build_mask(
                x, target=target)

            loss = 0

            # obj_conf
            loss_conf_obj = self.bce_loss(
                pred_obj_conf[obj_mask], t_obj_conf[obj_mask])
            loss_conf_noobj = self.bce_loss(
                pred_obj_conf[noobj_mask], t_obj_conf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            # class
            loss_cls = self.bce_loss(pred_class[obj_mask], tcls[obj_mask])



            loss_x = self.mse_loss(tx[obj_mask], x[obj_mask])
            loss_y = self.mse_loss(ty[obj_mask], y[obj_mask])
            loss_w = self.mse_loss(tw[obj_mask], w[obj_mask])
            loss_h = self.mse_loss(th[obj_mask], h[obj_mask])



            loss = loss_x+loss_y+loss_w+loss_h + loss_conf+loss_cls

            return outputs, loss


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

    x[:, :, 0] = torch.sigmoid(x[:, :, 0])  # x
    x[:, :, 1] = torch.sigmoid(x[:, :, 1])  # y
    x[:, :, 4:] = torch.sigmoid(x[:, :, 4:])  # obj and class

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    anchors = torch.FloatTensor(anchors)
    if use_cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        anchors = anchors.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, len(anchors)).view(-1, 2).unsqueeze(0)

    x[:, :, :2] += x_y_offset

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * anchors
    x[:, :, :2] *= stride
    return x


if __name__ == "__main__":
    def test_cal_iou_wh():
        anchors = [torch.FloatTensor([10, 13]), torch.FloatTensor([14, 15])]
        gwh = torch.FloatTensor([[10, 13], [11, 14], [10, 13]])
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
        print(ious.max(dim=0))

    def test_slices():
        a = torch.LongTensor([1, 0, 1, 1, 2])  # !!attention: 可以进行多次引用！！！
        b = torch.FloatTensor([0.1, 0.2, 0.3])
        for item in b:
            print(item)

    def test_grid():
# grid_x:
#     tensor([[[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#             [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]]]])
# grid_y:
#     tensor([[[[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#             [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
#             [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
#             [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
#             [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
#             [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
#             [ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
#             [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7],
#             [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
#             [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9],
#             [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
#             [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
#             [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]]]]) 
        grid_x = torch.arange(13).repeat(13, 1).view([1, 1, 13, 13])
        grid_y = torch.arange(13).repeat(13, 1).t().view([1, 1, 13, 13])
        print(grid_x)

    def test_cat():
        bb_targets = torch.FloatTensor([[1,2,3,4]])
        print(torch.cat(bb_targets,0))

    # test_cal_iou_wh()
    # test_slices()
    # test_grid()
    test_cat()
