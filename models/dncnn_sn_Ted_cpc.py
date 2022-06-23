
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
import math

# from models.PnP.realSN import real_spectral_norm

class ScalarMultiplyLayer(nn.Module):
    def __init__(self, scalar=1):
        super(ScalarMultiplyLayer, self).__init__()
        self.scalar = scalar
        
    def forward(self, x):
        return self.scalar*x

    def extra_repr(self):
        return 'L={scalar}'.format(**self.__dict__)

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.bias = nn.Parameter(torch.zeros(num_features))

        # self.running_mean = torch.zeros(num_features)
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, inp):
        size = list(inp.size())
        beta = self.bias.view(1, self.num_features, 1, 1)

        if self.training:
            avg = torch.mean(inp, dim=3)
            avg = torch.mean(avg, dim=2)
            avg = torch.mean(avg, dim=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * avg
        else:
            # avg = self.running_mean.repeat(size[0], 1)
            avg = self.running_mean

        output = inp - avg.view(1, self.num_features, 1, 1)
        output = output + beta

        return output

    def extra_repr(self):
        return '{num_features}, momentum={momentum} '.format(**self.__dict__)

@register_model("dncnn_sn_Ted_cpc")
class DnCNN_SN_TED_CPC(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=3, bnorm_type='mean', snorm=True, realsnorm=False, kernel_size=3, padding=1, L=1, n_power_iterations=5,residual=True, bias = True):
        super(DnCNN_SN_TED_CPC, self).__init__()
        self.residual = residual
        hidden_bias = not ((bnorm_type == 'full') or (bnorm_type == 'mean'))
        layers = []
        L_layer = L
        conv = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        if snorm:
            layers.append(nn.utils.spectral_norm(conv, n_power_iterations=n_power_iterations))
            layers.append(ScalarMultiplyLayer(L_layer))
#         elif realsnorm:
#             layers.append(real_spectral_norm(conv,n_power_iterations=n_power_iterations))
#             layers.append(ScalarMultiplyLayer(L_layer))
        else:
            layers.append(conv)
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            # Add convolutional layer (with weight/ spectral normalization if required)
            conv_layer = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=hidden_bias)
            if snorm:
                layers.append(nn.utils.spectral_norm(conv_layer,n_power_iterations=n_power_iterations))
                layers.append(ScalarMultiplyLayer(L_layer))
#             elif realsnorm:
#                 layers.append(real_spectral_norm(conv_layer,n_power_iterations=n_power_iterations))
#                 layers.append(ScalarMultiplyLayer(L_layer))
            else:
                layers.append(conv_layer)
            # Add required batch norm
            if bnorm_type == 'full':
                layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            elif bnorm_type == 'mean':
                layers.append(MeanOnlyBatchNorm(n_channels,momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        conv = nn.Conv2d(in_channels=n_channels, out_channels=2, kernel_size=kernel_size, padding=padding, bias=False)
        if snorm:
            layers.append(nn.utils.spectral_norm(conv,n_power_iterations=n_power_iterations))
            layers.append(ScalarMultiplyLayer(L_layer))
#         elif realsnorm:
#             layers.append(real_spectral_norm(conv,n_power_iterations=n_power_iterations))
#             layers.append(ScalarMultiplyLayer(L_layer))
        else:
            layers.append(conv)
        self.DnCNN_SN_TED_CPC = nn.Sequential(*layers)
        self._initialize_weights()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--in-channels", type=int, default=3, help="number of channels")
        parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension")
        parser.add_argument("--num-layers", default=20, type=int, help="number of layers")
#         parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument('--L', type=float, default=1.0, help='Lipschitz constant of network')
        parser.add_argument('--snorm', action='store_true', help='Turns on spectral normalization')
        parser.add_argument('--realsnorm', action='store_true', help='Turns on real spectral normalization')
        
        
    @classmethod
    def build_model(cls, args):
        return cls(image_channels = args.in_channels, n_channels = args.hidden_size, depth = args.num_layers, L = args.L, snorm = args.snorm, realsnorm = args.realsnorm)
    
    
    def forward(self, x):
        y = x
        out = self.DnCNN_SN_TED_CPC(x)
        if self.residual:
            return y[:, 0:2, :,:]-out
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

