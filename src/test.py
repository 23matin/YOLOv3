from __future__ import division

import vs_common as vs
from darknet import *
from torchstat import stat

cfg_file = r"../cfg/yolov3.cfg"
weight_file = r'../data/weights/yolov3.weights'


def test_cfg():
    prase_cfg(cfg_file)


def testBlockCreater():
    bc = BlockCreater()
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
    for idx, aa, bb in enumerate(zip(a, b)):
        print(idx)
        print(aa, bb)


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


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def get_test_input(img_path=None):
    if img_path is None:
        img_path = ".../data/test_img/dog-cycle-car.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    # print(img)
    return img_


def test_net():
    weight_file = '.../data/weights/yolov3.weights'
    net = Darknet(cfg_file, use_cuda=False)
    net.load_weights(weight_file)
    img = get_test_input()
    pred = net(img)
    print(pred)
    print(pred.shape)


colors = pkl.load(open("../data/others/pallete", "rb"))
classes = load_classes('../data/others/coco.names')


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0} {1:.3f}".format(classes[cls], x[5]*x[6])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def test_predict_img(img_path=None,img_show = False):
    confidence = 0.5
    nms_thresh = 0.4
    inp_dim = 416
    use_cuda = 1 and torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_cuda:
        print("predict using cuda!")
    else:
        print("not using cuda!")
    if img_path is None:
        # img_path = '../data/img/dog-cycle-car.png'
        # img_path = '../data/img/bus.jpg'
        img_path = '../data/img/train.jpg'
    # weight_file = '../data/weights/yolov3.weights'
    net = Darknet(cfg_file, use_cuda=use_cuda)
    # print(net)
    net.load_weights(weight_file)
    net.eval()

    # see net param
    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    frame = cv2.imread(img_path)
    img = prep_image(frame, inp_dim)
    batch = 1
    img = torch.cat([img for i in range(batch)], 0)
    print(img)
    labels = torch.Tensor([[0, 0, 0.515, 0.5, 0.21694873, 0.18286777]])
    labels = torch.cat([labels for i in range(batch)], 0)
    if batch > 1:
        labels[:, 0] = torch.arange(batch)

    timer = vs.Timer()
    if use_cuda:
        img = img.cuda()
        net.cuda()
    with torch.no_grad():
        timer.start()
        out, loss = net(img, labels)
        timer.stop()
        print("predict a img used {} ms.".format(timer.getMsec()))
    output = write_results(out, confidence, len(classes), nms_thresh)

    if 1:
        path = "../data/weights/coco.pt"
        torch.save(net.state_dict(), path)

    output[:, [1, 3]], output[:, [2, 4]] = letterbox_convert_back(
        frame, (inp_dim, inp_dim), output[:, [1, 3]], output[:, [2, 4]])
    print("output:", output)
    print("loss:", loss)
    list(map(lambda x: write(x, frame), output))
    cv2.imwrite("frame.png", frame)
    if img_show:
        cv2.imshow("frame.png", frame)
        cv2.waitKey()


def test_cam():
    confidence = 0.5
    nms_thresh = 0.4
    inp_dim = 416

    weight_file = '../data/weights/coco.pt'
    weight_file = '../data/weights/yolov3.weights'
    use_cuda = 1 and torch.cuda.is_available()

    net = Darknet(cfg_file, use_cuda=use_cuda)
    net.load_weights(weight_file)
    net.eval()

    # ?????????????????????
    if False:
        net = Darknet(cfg_file)
        stat(net, (3, inp_dim, inp_dim))
        return

    if use_cuda:
        print("predict using cuda!")
        net = net.cuda()

    cap = cv2.VideoCapture(0)
    timer = vs.Timer()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            img = prep_image(frame, inp_dim)
            repeat = 1
            img = torch.cat([img for i in range(repeat)], 0)
            if use_cuda:
                img = img.cuda()
            timer.start()
            out,_ = net(img)
            timer.stop()
            print("predict a img used {} ms.".format(timer.getMsec()/repeat))
            output = write_results(out, confidence, len(classes), nms_thresh)
            if output is not None:
                output[:, [1, 3]], output[:, [2, 4]] = letterbox_convert_back(
                    frame, (inp_dim, inp_dim), output[:, [1, 3]], output[:, [2, 4]])
                list(map(lambda x: write(x, frame), output))
            cv2.imshow("frame", frame)
            cv2.waitKey(1)


def test_timer():
    t = Timer()
    time.sleep(1)
    t.stop()
    print(t.getSec())


def test_import():
    import os
    os.chdir("/home/matin23")
    # os.system("ls")
    path = "/home/matin23/PycharmProjects/yolov3/src"
    import sys
    sys.path.append(path)
    import vs_common
    dir(vs_common)
    a = vs_common.Timer()
    print(a)


def test_nettron():
    import netron
    netron.start("/home/matin23/workspace/my_yolov3/data/weights/coco.pt")


def test_tensorboard():
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from torchviz import make_dot
    import torchvision.models as models

    cfg_file = "/home/matin23/workspace/my_yolov3/cfg/yolov3.cfg"
    net = Darknet(cfg_file)

    fake_img = torch.rand(1, 3, 416, 416)  # ?????????????????????
    out = net(fake_img)
    # print(out)

    # 1. ??????tensorboarf???????????????
    writer = SummaryWriter(comment='test_your_comment',
                           filename_suffix="_test_your_filename_suffix")
    writer.add_graph(net, fake_img)  # ???????????????????????????

    # 2. ?????????pt????????????????????????
    torch.save(net, "alexnet.pt")

    # 3. ??????graphviz???????????????
    out = net(fake_img)
    g = make_dot(out)
    g.render('alexnet', view=True)  # ???????????????????????????pdf??????


def test_ConvBnBlock():
    # inp, oup, enable_bn, actv_fn, kernel_size, stride, pad, idx=-1)
    block = ConvBnBlock(10, 10, 1, 'relu', 3, 1, 1)
    print(block)


def test_netron():
    onnx_path = "onnx_model_name.onnx"
    net = Darknet(cfg_file)
    net.load_state_dict(torch.load(weight_pt))
    d = torch.rand(1, 3, 416, 416)
    o = net(d)
    torch.onnx.export(net, d, onnx_path, opset_version=11)
    import netron
    netron.start(onnx_path)

def test_simple():
    # test_cfg()
    # testBlockCreater()
    # test_dict()
    # test_str2int_or_float()
    # test_list()
    # get_test_input()
    # test_net()
    test_predict_img()
    # test_timer()
    # test_import()
    # test_cam()
    # test_nettron()
    # test_tensorboard()
    # test_ConvBnBlock()
    # test_netron()

test_simple()
