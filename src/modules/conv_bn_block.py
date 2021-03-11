import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvBnBlock(nn.Module):
    def __init__(self, inp, oup, enable_bn, actv_fn, kernel_size, stride, pad, idx=-1):
        super(ConvBnBlock, self).__init__()
        if int(pad):
            pad = kernel_size // 2
        else:
            pad = 0

        self.conv = nn.Conv2d(
            inp, oup, kernel_size, stride, pad, bias=(not enable_bn))
        self.bn = nn.BatchNorm2d(oup) if enable_bn else None
        self.act = None

        if actv_fn == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif actv_fn == "relu":
            self.act = nn.ReLU(inplace=True)
        elif actv_fn == "linear":
            pass
        else:
            raise Exception(
                "activation type '{}' is not support for now!".format(actv_fn))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if actv_fn is not None:
            x = self.act(x)
        return x


def make_conv_bn(inp, oup, enable_bn, actv_fn, kernel_size, stride, pad, idx=-1):
    """
    make a conv bn layer.
    inp: input channels
    oup: output channels
    enable_bn: wehther to have bn, if not bn, conv要加上bias
    actv_fn: activate function :["leaky", "relu", "linear"]
    kernel_size: kernel_size
    pad: wether to pad 'kernel // 2'
    idx： if idx >=0 will output something like :
        'Sequential(
            (conv_10): Conv2d(100, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn_10): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu_10): ReLU(inplace=True)
        )'
    if idx <0 will output something like:
        Sequential(
            (0): Conv2d(100, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
        )
    """
    if int(pad):
        pad = kernel_size // 2
    else:
        pad = 0
    conv_module = nn.Conv2d(
        inp, oup, kernel_size, stride, pad, bias=(not enable_bn))
    bn_module = nn.BatchNorm2d(oup) if enable_bn else None
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
    seq = nn.Sequential()
    if idx >= 0:
        seq.add_module("conv_{}".format(idx), conv_module)
        if bn_module:
            seq.add_module("bn_{}".format(idx), bn_module)
        if activation_module:
            seq.add_module("{}_{}".format(
                actv_fn, idx), activation_module)
    else:
        module_list = []
        module_list.append(conv_module)
        if bn_module:
            module_list.append(bn_module)
        if activation_module:
            module_list.append(activation_module)
        seq = nn.Sequential(*module_list)
    return seq


if __name__ == 'main':
    print(make_conv_bn(100, 100, 1, "relu", 3, 1, 1))
    print(make_conv_bn(100, 100, 1, "relu", 3, 1, 1, 100))
