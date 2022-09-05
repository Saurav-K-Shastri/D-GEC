import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch

from algorithm import dvdamp
from algorithm import simulation as sim

from algorithm import heatmap
from utils import general as gutil
from utils import plot as putil
from utils import transform as tutil
from utils import my_transforms as mutil
from utils.general import load_checkpoint, save_image
from utils import GEC_pytorch_denoiser_util as dutils
from utils import *

from fastMRI_utils import transforms_new

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

import utils.wave_torch_transforms as wutils

from numpy import linalg as LA

class A_op_multi:
    
    def __init__(self,idx1_complement,idx2_complement, sens_map):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sens_map = sens_map
        self.nc = sens_map.shape[0]
        
    def A(self,X):
        X = (X).permute(0,2,3,1)
        X_sens = transforms_new.complex_mult(X.unsqueeze(1).repeat(1,self.nc,1,1,1), self.sens_map.unsqueeze(0))
        out_foo = transforms_new.fft2c_new(X_sens)
        out_foo[:,:,self.idx1_complement,self.idx2_complement,:] = 0
        out = torch.cat((out_foo[:,:,:,:,0], out_foo[:,:,:,:,1]), dim = 1)
        return out

    def H(self,X):
        X = X.permute(0,2,3,1)
        X[:,self.idx1_complement,self.idx2_complement,:] = 0
        X_new = torch.stack([X[:,:,:,0:self.nc],X[:,:,:,self.nc:]], dim = -1).permute(0,3,1,2,4)
        out_sens = transforms_new.ifft2c_new(X_new)
        out_foo = torch.sum(transforms_new.complex_mult(out_sens, transforms_new.complex_conj(self.sens_map.unsqueeze(0))),dim = 1)
        out = out_foo.permute(0,3,1,2)
       
        return out
    

def find_spec_rad(mri,steps, n, device):
    # init x
    x = torch.randn((1,2,n,n), device = device)
    x = x/torch.sqrt(torch.sum(transforms_new.complex_abs(x.permute(0,2,3,1))**2))

    # power iteration
    for i in range(steps):
        x = mri.H(mri.A(x))
        spec_rad = torch.sqrt(torch.sum(transforms_new.complex_abs(x.permute(0,2,3,1))**2))
        x = x/spec_rad
        
    return spec_rad


def PnP_PDS(y, sens_maps_new, mask, wvar, num_of_PnP_PDS_iterations, trained_DnCNN_white, gamma_tune, GT_target, metric_mask):
    
    
    model_pnp = load_model(trained_DnCNN_white)
    device = y.device
    scale_percentile = 98
    
    model_pnp.to(device=device)
    model_pnp.eval()

    idx1_complement = np.where(mask == 0)[0]
    idx2_complement = np.where(mask == 0)[1]
    idx1 = np.where(mask == 1)[0]
    idx2 = np.where(mask == 1)[1]
    
    EYE = torch.ones_like(y)

    mri = A_op_multi(idx1_complement,idx2_complement,sens_maps_new)

    alpha = 0
    
    n = y.shape[-1]
        
    PSNR_list_PDS = []

                
    with torch.no_grad():

        x = mri.H(y)
        
        recovered_image_PNP = transforms_new.complex_abs(x.squeeze(0).permute(1,2,0))
#         PSNR_list_PDS.append(gutil.calc_psnr((recovered_image_PNP*metric_mask).cpu(), (GT_target*metric_mask).cpu(), max = (GT_target*metric_mask).max().cpu()))
            
        z = torch.zeros_like(y)
        
        Ht_y = mri.H(y);
        L = find_spec_rad(mri,10, n, device)

        gamma_1 = gamma_tune
        gamma_2 = (1/L)*((1/gamma_1))

        for k in range(num_of_PnP_PDS_iterations):

            yoda = 1

            b1 = x - (gamma_1*mri.H(z))

            sorted_image_vec = transforms_new.complex_abs(b1.squeeze(0).permute(1,2,0)).reshape((-1,)).sort()
            scale = sorted_image_vec.values[int(len(sorted_image_vec.values) * scale_percentile/100)].item() # Because Denoiser is trained for such a scaling. Scaling not necessary if denoiser is Bias-Free
        
            scaled_b1 = b1/scale
            
            denoised_scaled = model_pnp(scaled_b1)
            denoised = scale*denoised_scaled

            x_new = denoised.clone()

            qoo = (mri.A(x_new) - y)

            too = torch.sum((qoo.permute(0,2,3,1))**2)

            x_hat = x_new + 1*(x_new - x)

            z = (1/(1 + gamma_2))*z + (gamma_2/(1+gamma_2))*(mri.A(x_hat) - y)

            x = x_new.clone()

            boo = mri.A(x) - y

#             resnorm_recov = torch.sqrt(torch.sum((boo.permute(0,2,3,1))**2)).cpu()
            recovered_image_PNP = transforms_new.complex_abs(x.squeeze(0).permute(1,2,0))
            PSNR_list_PDS.append(gutil.calc_psnr((recovered_image_PNP*metric_mask).cpu(), (GT_target*metric_mask).cpu(), max = (GT_target*metric_mask).max().cpu()))
            
#             x_iter_mat[k,:,:,:,:] = x


    return x, PSNR_list_PDS


