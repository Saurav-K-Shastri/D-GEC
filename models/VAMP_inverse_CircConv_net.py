"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# author:Saurav (shastri.19@osu.edu)

import torch
from torch import nn
from torch.nn import functional as F
from models import register_model

@register_model("VAMP_inverse_CircConv_net")
class VAMP_inverse_CircConv_net(nn.Module):


    def __init__(self, in_chans=2, out_chans=2, image_height = 320, bias=False):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            image_height: assuming square image
            bias: True or False
        """
        super(VAMP_inverse_CircConv_net, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.bias = bias
        self.image_height = image_height
    
        self.skip_connection_weight = torch.nn.Parameter(1*torch.ones(()))
        
        self.first_layer = nn.Conv2d(in_channels=in_chans, out_channels=out_chans,
                                     kernel_size=image_height, padding=1, bias=self.bias)
        
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--in-channels", type=int, default=2, help="Number of channels in the input to CirConvNet model")
        parser.add_argument("--out-channels", type=int, default=2, help="Number of channels in the output to CirConvNet model")
        parser.add_argument("--image-height", type=int, default=320, help="height of the image assuming square image i.e. height = width")
#         parser.add_argument("--residual-connection", action='store_true', help="if residual connection required")
        parser.add_argument("--bias", action='store_true', help="this is not used anywhere significant. It is used in file names ")
    
    @classmethod
    def build_model(cls,args):
        return cls(in_chans = args.in_channels, out_chans = args.out_channels, image_height = args.image_height, bias=args.bias)
        

    def forward(self, x):
        y = x
        x = F.pad(x,(self.image_height,self.image_height,self.image_height,self.image_height),"circular")
        out = self.first_layer(x)
        
        out = out[:,:,2:2+self.image_height,2:2+self.image_height] # accessing the circular convolution result

        
        out = self.skip_connection_weight*y + out
#         output = self.skip_connection_weight*image
            
            
        return out
