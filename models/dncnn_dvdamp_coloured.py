import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
from models import BFBatchNorm2d
import math
import torch

@register_model("dncnn_dvdamp_coloured")
class DnCNN__dvdamp_coloured(nn.Module):
    """Made a modification to DnCNN (as defined in https://arxiv.org/abs/1608.03981
    reference implementation: https://github.com/SaoYan/DnCNN-PyTorch) to mimic Prof. Metzler's coloured denoiser proposed in DVDAMP paper (https://arxiv.org/pdf/2010.13211.pdf)"""
    def __init__(self, depth=20, n_channels=64, image_channels=1, std_channels = 13, bias=False, kernel_size=3):
        super(DnCNN__dvdamp_coloured, self).__init__()
        kernel_size = 3
        padding = 1

        self.bias = bias
        if not bias:
            norm_layer = BFBatchNorm2d.BFBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.depth = depth

        self.first_layer = nn.Conv2d(in_channels=image_channels+std_channels, out_channels=n_channels,
                                     kernel_size=kernel_size, padding=padding, bias=self.bias)

        self.hidden_layer_list = [None] * (self.depth - 2)

        self.bn_layer_list = [None] * (self.depth - 2)

        for i in range(self.depth-2):
            self.hidden_layer_list[i] = nn.Conv2d(
                in_channels=n_channels+std_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)
            self.bn_layer_list[i] = norm_layer(n_channels)

        self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list)
        self.bn_layer_list = nn.ModuleList(self.bn_layer_list)
        self.last_layer = nn.Conv2d(in_channels=n_channels+std_channels, out_channels=image_channels,kernel_size=kernel_size, padding=padding, bias=self.bias)  # modified wrt to regular dncnn

        self._initialize_weights()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--in-channels", type=int,
                            default=1, help="number of channels")
        parser.add_argument("--std-channels", type=int,
                            default=13, help="number of std channels")
        parser.add_argument("--hidden-size", type=int,
                            default=64, help="hidden dimension")
        parser.add_argument("--num-layers", default=20,
                            type=int, help="number of layers")
        parser.add_argument("--bias", action='store_true',
                            help="use residual bias")

    @classmethod
    def build_model(cls, args):
        return cls(image_channels=args.in_channels,std_channels = args.std_channels, n_channels = args.hidden_size, depth = args.num_layers, bias=args.bias)


    def forward(self, x, std):
        _, _, H, W = x.shape
        std_channels = self._generate_std_channels(std, H, W)
        noise = torch.cat((x, std_channels), dim=1)
        
        noise = self.first_layer(noise)
        noise = F.relu(noise)

        for i in range(self.depth-2):
            noise = torch.cat((noise, std_channels), dim=1)
            noise = self.hidden_layer_list[i](noise)
            noise = self.bn_layer_list[i](noise)
            noise = F.relu(noise)
        noise = torch.cat((noise, std_channels), dim=1)
        noise = self.last_layer(noise)
        out = x - noise

        return out


    def _generate_std_channels(self, std, H, W):
        N, concat_channels = std.shape
        std_channels = std.reshape(N, concat_channels, 1, 1).repeat(1, 1, H, W)
        return std_channels


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m,BFBatchNorm2d.BFBatchNorm2d):
                m.weight.data.normal_(mean=0, std=math.sqrt(
                    2./9./64.)).clamp_(-0.025, 0.025)
                init.constant_(m.bias, 0)
