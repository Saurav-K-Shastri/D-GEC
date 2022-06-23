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

from models.model import Colored_DnCNN 


def D_VDAMP(y, mask, prob_map, wvar, num_of_D_VDAMP_iterations, modelnames, modeldir, GT_target):
    
    device = y.device
    level = 4
    wavelet = 'haar'
    y_dummy = (y.clone()).permute(0,2,3,1).squeeze(0).cpu().numpy()
    y_np = y_dummy[:,:,0] + 1j*y_dummy[:,:,1]

    var0_dvdamp = wvar.cpu().numpy()
    
    oneim = True
    verbose = False
    dentype = 'cdncnn' # 'bm3d' choices=['soft', 'bm3d', 'cdncnn']

    stop_on_increase = False

    std_ranges = np.array([0, 10, 20, 50, 120, 500]) / 255

    modeldirs = [None] * len(modelnames)

    std_channels = 3 * level + 1
    for i, modelname in enumerate(modelnames):
        modeldirs[i] = os.path.join(modeldir, '{}.pth'.format(modelname))
        
    denoiser = dvdamp.ColoredDnCNN_VDAMP(modeldirs, std_ranges, std_channels=std_channels, beta_tune = 1, device=device, verbose=verbose)

    x_hat_dvdamp, x_hat_dvdamp_mat, log_dvdamp, true_iters_dvdamp = dvdamp.dvdamp2(y_np, prob_map, mask, var0_dvdamp, denoiser,
                                    image=None, iters=num_of_D_VDAMP_iterations, level=level, 
                                    wavetype=wavelet, stop_on_increase=stop_on_increase)
    
    recon_image_dvdamp = gutil.im_numpy_to_tensor(x_hat_dvdamp)
    recon_image_dvdamp_mat = gutil.im_numpy_to_tensor(x_hat_dvdamp_mat)
    
#     print(recon_image_dvdamp_mat.shape)
#     print(x_hat_dvdamp_mat.shape)
#     print(y.shape)
    
    recon_dvdamp = torch.zeros_like(y)
    recon_dvdamp[:,0,:,:] = torch.real(recon_image_dvdamp) 
    recon_dvdamp[:,1,:,:] = torch.imag(recon_image_dvdamp) 
    PSNR_list_DAMP = []
    for i in range(x_hat_dvdamp_mat.shape[0]):
        
            recon_dvdamp_dummy = torch.zeros_like(y)
            recon_dvdamp_dummy[:,0,:,:] = torch.real(recon_image_dvdamp_mat[0,i,:,:].to(device))
            recon_dvdamp_dummy[:,1,:,:] = torch.imag(recon_image_dvdamp_mat[0,i,:,:].to(device))
            
            PSNR_list_DAMP.append(gutil.calc_psnr((recon_dvdamp_dummy[0,0,:,:]).cpu(), (GT_target[0,0,:,:]).cpu(), max = (GT_target[0,0,:,:]).max().cpu()))
            
    return recon_dvdamp,PSNR_list_DAMP
