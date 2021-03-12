import os

from utils import *

def to_cpu(tensor):
    return tensor.detach().cpu()

class Darknet(nn.Module):
    def __init__(self, cfg_file, use_cuda=False):
        super(Darknet, self).__init__()
        self.use_cuda = use_cuda
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

    def forward(self, x, target=None):
        outputs = dict()
        output_filters = []
        output_filters.append(x.size(1))
        detections = None  # accumulate decoded detections
        loss = 0
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
                else:
                    concate_list = list()
                    for route_idx in merge_layers:
                        if route_idx > 0:
                            route_idx -= idx
                        concate_list.append(outputs[idx + route_idx])
                    x = torch.cat(concate_list, dim=1)
            elif module_type == "yolo":
                x, layer_loss = module[0](x, target, use_cuda=self.use_cuda)
                loss += layer_loss
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[idx] = x
            if False:
                print("type: ", module_type)
                print("outputs[{}]".format(idx), outputs[idx])
        detections = to_cpu(detections)
        return detections, loss

    def load_weights(self, weightfile):
        if os.path.splitext(weightfile)[-1] == '.pt':
            self.load_state_dict(torch.load(weightfile))
            return

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
