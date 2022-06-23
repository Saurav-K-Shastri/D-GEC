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

class A_op:
    
    def __init__(self,idx1_complement,idx2_complement):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        
    def A(self,X):
        out = transforms_new.fft2c_new(X.permute(0,2,3,1))
        out[:,self.idx1_complement,self.idx2_complement,:] = 0
        return out.permute(0,3,1,2)

    def H(self,X):
        X = X.permute(0,2,3,1)
        X[:,self.idx1_complement,self.idx2_complement,:] = 0
        out1 = transforms_new.ifft2c_new(X)
        out = out1.permute(0,3,1,2)
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


def PnP_PDS(y, mask, wvar, num_of_PnP_PDS_iterations, trained_DnCNN_white, gamma_tune,GT_target):
    
    
    model_pnp = load_model(trained_DnCNN_white)
    device = y.device
    
    model_pnp.to(device=device)
    model_pnp.eval()

    idx1_complement = np.where(mask == 0)[0]
    idx2_complement = np.where(mask == 0)[1]
    idx1 = np.where(mask == 1)[0]
    idx2 = np.where(mask == 1)[1]
    
    EYE = torch.ones_like(y)

    mri = A_op(idx1_complement,idx2_complement)

    alpha = 0
    
    n = y.shape[-1]
    PSNR_list_PDS = []
    
    with torch.no_grad():

        x = mri.H(y)
        z = torch.zeros_like(y)
        
        Ht_y = mri.H(y);
        L = find_spec_rad(mri,10, n, device)

        gamma_1 = gamma_tune
        gamma_2 = (1/L)*((1/gamma_1))

        for k in range(num_of_PnP_PDS_iterations):

            yoda = 1

            b1 = x - (gamma_1*mri.H(z))

            denoised_real = model_pnp(b1[:,0,:,:].unsqueeze(0))
            denoised = torch.cat([denoised_real,torch.zeros_like(denoised_real)],dim = 1)

            x_new = denoised.clone()

            qoo = (mri.A(x_new) - y)

            too = torch.sum((qoo.permute(0,2,3,1))**2)

            x_hat = x_new + 1*(x_new - x)

            z = (1/(1 + gamma_2))*z + (gamma_2/(1+gamma_2))*(mri.A(x_hat) - y)

            x = x_new.clone()

            boo = mri.A(x) - y

            resnorm_recov = torch.sqrt(torch.sum((boo.permute(0,2,3,1))**2)).cpu()

#             recovered_image_PNP = transforms_new.complex_abs(x.squeeze(0).permute(1,2,0))
            recovered_image_PNP = x.clone()
    
            PSNR_list_PDS.append(gutil.calc_psnr((recovered_image_PNP[0,0,:,:]).cpu(), (GT_target[0,0,:,:]).cpu(), max = (GT_target[0,0,:,:]).max().cpu()))
    

    return x, PSNR_list_PDS


