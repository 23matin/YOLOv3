# YOLOv3 复现

This reposity mainly used to code YOLOv3 and hope I can learn something in it.

## Refs

[1] [How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

## TODOs

- [x] Code refactoring
- [x] Cuda support
- [ ] Train support
- [ ] 测试集评价指标
- [ ] form a python version of lib 'vs_common'

**# YOLOv3 复现**



This reposity mainly used to code YOLOv3 and hope I can learn something in it.



**## Refs**



[1] [How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)



**## TODOs**



\- [x] Code refactoring

\- [x] Cuda support

\- [ ] Train support

\- [ ] 测试集评价指标

\- [ ] form a python version of lib 'vs_common'



##  output 解码过程



1. 首先output是的size是 det of size $(N_{batch}, N_{anchors} \times N_{grid} \times N_{grid}, N_{classes}+5)$

2. 根据论文里面对应的过程:

​    假设输出是```det```, 首先将预测转换为：

​    $(N_{batch}, N_{anchors} , N_{grid} , N_{grid}, N_{classes}+5)$

​    得到预测的```obj_conf```,所有预测的```coordinate(px,py,pd,pw)``` ,```class_conf```。

3. 对置信度进行sigmoid变换

4. 对x和y进行和缩放和位移变换以及对w和h进行变换

5. 得到最终的输出

代码对应为：

```python
#从det中分离三类预测变量
obj_conf=det[...,4]      #of shape [n_batch,n_anchors,n_grid,n_grid,1]
coordinate = det[...,:4] #of shape [n_batch,n_anchors,n_grid,n_grid,4]
class_conf = det[...,5:] #of shape [n_batch,n_anchors,n_grid,n_grid,n_classes]
px,py,pw,ph = coordinate[...,0],coordinate[...,1],coordinate[...,2],coordinate[...,3]

#对置信度变换
obj_conf = torch.sigmoid(obj_conf)
class_conf = torch.sigmoid(class_conf)

#对x和y进行sigmoid变换到(0，1)后再将对应格子的左上角的坐标加上去
px,py = torch.sigmiod(px),torch.exp(py)

#img_size
#n_grid

offset_x = torch.arange(n_grid).repeat(n_grid,1).view(1,1,n_grid,n_grid)
offset_y = torch.arange(n_grid).repeat(n_grid,1).t().view(1,1,n_grid,n_grid)

px = px + offset_x
py = py + offset_y

px *= img_size/m_grid #->转换到原图坐标系上
py *= img_size/m_grid

# w 和 h 先 指数变换再乘以对应anchor的长宽
#假设对应的anchors 是变量anchors of shape (n_anchors,2)
default_anchor_w = anchors[:,0].view(1,n_anchors,1,1,1)
default_anchor_h = anchors[:,0].view(1,n_anchors,1,1,1)

"""
yolo3 中的默认框是相对于默认输入的，单位是像素
比如，最小的默认框10,13 代表416尺寸大小的图像对应的最小的默认框是10，13

而预测的中心位移是相对于左上角的比例
预测的缩放是对应anchor的比例
因此有一个不同分辨率图像的转换的一个过程
"""

pw = torch.exp(pw) * default_anchor_w
ph = torch.exp(ph) * default_anchor_h

# 得到最终的输出
output = torch.cat((px,py,pw,ph,obj_conf,class_conf),-1)
```

## 一些思考：
正负样本分配的过程中，如果不考虑正样本重叠的情况，(即求解最大iou的时候，多个gtbox对应于同一个anchor), 那么每个gtbox必产生3个正样本，即便是iou很小，比如：
```
best_ious, tensor([0.6576])
best_ious, tensor([0.4776])
best_ious, tensor([0.1106])
```
虽然iou很小但是还是强行分配为正样本会导致可能根本没有合适的特征去预测对应的目标，而会导致loss很大。一个可能比较好的方法是:维持当前计算方法，但是对iou特别小的置为负样本
比如上面最后一个层次的最大的iou才只有0.11，因为目标太小了按照负样本的生成规则(thresdhold < 0.5)应该生成为负样本，但是却是正样本，有一些矛盾



## loss 计算过程

### loss的组成

loss 主要有两大部分组成：

1. 正样本loss 

   ​	正样本loss包括：

   1. obj_conf_loss (有没有目标的置信度，真值为1)
   2. Class_conf_loss (分类loss)
   3. coord_loss(loss_x,loss_y,loss_w,loss_h) (定位loss)

2. 负样本loss.
   1. 负样本loss就只有noob_conf_loss(有没有目标的置信度，真值为0)

### 正样本和负样本的确定

yolo3中正样本和负样本的确定规则：

1. 将gt_box 和所有的anchor进行比较，iou最大的那个anchor为正样本
2. iou小于阈值的(0.5)就是负样本
3. 分别生成正样本和负样本的mask

要根据正负样本来确定哪一个anchorsbox是证样本哪一个是负样本，然后再进行loss的计算



代码如下：

```python
def gt_transform(gt,anchors):
  BoolTensor = torch.BoolTensor
  FloatTendor = torch.FloatTensor

  # 假设label为 gt of shape (n_gt,6),6个变量分别为：batch_idx , class_label, x,y,w,h
  gt_boxes = gt[..., 2:]
  gt_x,gt_y = gt_boxes[:,0],gt_boxes[:,1]
  gt_w,gt_h = gt_boxes[:,2],gt_boxes[:,3]

  obj_mask = BoolTensor(n_batch,n_anchors,n_grid,n_grid).fill_(False)
  noobj_mask = BoolTensor(n_batch,n_anchors,n_grid,n_grid).fill_(True)

  ious = calculate_iou_anchor_type(gt,anchors) #计算每一个gt_boxes和所有anchor形状的iou of shape (len(gt),len(anchors)) 这里存在一个问题是 iou最大的anchor可能也很小，因为默认的框和gt的尺寸差异可能本来就很大，所以一个框可能即是正样本又是负样本

  # 得到最大iou的anchor的idx
  best_iou, anchors_id = ious.argmax(0)

  #得到gt_box对应的grid坐标
  grid_x = (gt_x * grid_size)
  grid_y = (gt_y * grid_size)
	
  grid_x_idx = grid_x.long()
  grid_y_idx = grid_y.long()
  
  #生成 obj_mask
  batch_idx = gt[:,0]
  obj_mask[batch_idx,anchors_id,grid_x_idx,grid_y_idx] = True #batch_idx,anchors_id,grid_x_idx,grid_y 每一个都是len(gt)纬的响亮

  #生成 noobj_mask, iou大于iou_thres的不是负样本,所以默认很多负样本
  noobj_mask[batch_idx，ious>iou_thres，grid_x_idx,grid_y_idx] = False


  #得到对应的真值tx ty w h 
  tx = (grid_x - grid_x.floor())/stride
  ty = (grid_y - grid_y.floor())/stride

  #gt的w和h是相对于原图的，要把他们转换为相对于anchor的,再取对数
  tw = torch.log((gt_w*img_size)/anchor[anchors_id,0])
  th = torch.log((gt_h*img_size)/anchor[anchors_id,1])

  #生成t_cls
	t_cls = FloatTensor(gt.size(0),n_classes).fill_(0)
  t_cls[:,gt[:,1]] = 1
  
  
  return obj_mask,noobj_mask,tx,ty,tw,th,t_cls
```



### loss计算

有了 obj_mask,noobj_mask,tx,ty,tw,th，就直接可以进行loss的计算.代码如下：

```
obj_mask,noobj_mask,tx,ty,tw,th,tcls = gt_transform(gt,anchors)

# 正样本
obj_conf_loss = nn.BCELoss()(det[obj_mask,4],True) #有无目标
cls_conf_loss = nn.BCELoss()(det[obj_mask,5:],t_cls) #分类
coord_loss = nn.MSELoss()(det[obj_mask,:4],torch.cat((tx,ty,tw,th),1))

#负样本:
noobj_conf_loss = nn.BCELoss()(det[obj_mask,4],False)

#加权
loss = obj_scale * obj_conf_loss + noobj_scale * noobj_conf_loss + cls_conf_loss + noobj_conf_loss
```



