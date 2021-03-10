import torch
import torch.nn as nn
import torch.nn.functional as F

class EmptyLayer(nn.Module):
    """
    mainly use to form shortcut or rotute layers do no calculations
    """

    def __init__(self, layer_type, data=None):
        super(EmptyLayer, self).__init__()

        self.type = layer_type
        self.data = data
