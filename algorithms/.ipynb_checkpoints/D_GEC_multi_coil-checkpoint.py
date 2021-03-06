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

import scipy.misc

import cv2
import sigpy as sp
import sigpy.mri as mr

from fastMRI_utils import transforms_new

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

import utils.wave_torch_transforms as wutils

from numpy import linalg as LA
from scipy.io import loadmat


class B_op_multi:
    
    def __init__(self,idx1_complement,idx2_complement,xfm,ifm, level, sens_map):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.xfm = xfm
        self.ifm= ifm
        self.level = level 
        self.sens_map = sens_map
        self.nc = sens_map.shape[0]
        
    def A(self,X):
        X1 = (wutils.wave_inverse_mat(X,self.ifm,self.level)).permute(0,2,3,1)
        X_sens = transforms_new.complex_mult(X1.unsqueeze(1).repeat(1,self.nc,1,1,1), self.sens_map.unsqueeze(0))
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
        out_foo = out_foo.permute(0,3,1,2)
        out = wutils.wave_forward_mat(out_foo,self.xfm)
        
        return out



def D_GEC(y, sens_maps_new, mask, wvar, sens_var, num_of_D_GEC_iterations, modelnames, modeldir,theta_damp,zeta_damp, GT_target, metric_mask):

    wavelet = 'haar'
    level = 4
    num_of_sbs = 3*level + 1
    
    use_DDVAMP_damping = True
    num_of_CG_iter = 20
    scale_percentile = 98
    
    sigma_w_new = torch.sqrt(wvar + sens_var)
    
    device = y.device

    warm_start_for_LMSE =  True
    warm_start_vec = torch.zeros(1,2,y.shape[2],y.shape[3]).to(device)
    warm_start_mat = torch.zeros(3*level + 2,2,y.shape[2],y.shape[3]).to(device)
    
    xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)
       
    idx1_complement = np.where(mask == 0)[0]
    idx2_complement = np.where(mask == 0)[1]
    idx1 = np.where(mask == 1)[0]
    idx2 = np.where(mask == 1)[1]
    
    B_op_foo = B_op_multi(idx1_complement,idx2_complement,xfm,ifm, level, sens_maps_new)

    r1_bar_t = B_op_foo.H(y)
    
    GAMMA_1_full = (100000000*torch.rand((1,num_of_sbs),device= device)) # This initialization works too. Preferably use the method described in our paper. Our algorithm is quite robust to initialization.
    
    GAMMA_2_full = GAMMA_1_full.clone()
    
    p1m1_mask, subband_sizes = wutils.get_p1m1_mask_and_subband_sizes(torch.zeros_like(r1_bar_t),level)
    p1m1_mask = p1m1_mask.to(device)
    subband_sizes = subband_sizes.to(device)


    std_ranges = torch.tensor(np.array([0, 10, 20, 50, 120, 500]) / 255).to(device)
    
    modeldirs = [None] * len(modelnames)
    std_channels = 3 * level + 1
    
    verbose_req = False
    for i, modelname in enumerate(modelnames):
        modeldirs[i] = os.path.join(modeldir, '{}'.format(modelname))

    # DnCNN cpc Denoiser
    ColoredDnCNN_GEC_denoiser_complex = dutils.DnCNN_cpc_VDAMP_true_complex_batch(modeldirs, std_ranges, xfm,ifm, p1m1_mask, subband_sizes, std_channels=std_channels,  beta_tune = torch.tensor(1).to(device), device=device, verbose=verbose_req, level = level,scale_percentile = scale_percentile )

    #LMMSE Denoiser
    LMSE_denoiser_CG_warm = dutils.LMSE_batch_CG_GEC_with_div_and_warm_strt_multi_coil(y, idx1_complement, idx2_complement, sigma_w_new, xfm, ifm, p1m1_mask, subband_sizes, sens_maps_new, 4, num_of_CG_iter,torch.tensor(1), torch.tensor(1e-4))

    PSNR_list = []
    
    
    with torch.no_grad():
        
        for iter in range(num_of_D_GEC_iterations):            
            
            # Linear Stage
            x_1_bar_cap_t, D1_t, warm_start_mat = LMSE_denoiser_CG_warm(r1_bar_t, (1/GAMMA_1_full),warm_start_mat, True)

        
            D1_t = D1_t.reshape(1,3*level + 1)
            D1_t = torch.abs(D1_t)
            
            recovered_image_LMMSE = wutils.wave_inverse_mat(x_1_bar_cap_t.clone(),ifm, level)

            if iter>0:
                r2_bar_t_old = r2_bar_t.clone()
                GAMMA_2_full_old = GAMMA_2_full.clone()

            fac1 = 1/(1 - D1_t)

            r2_bar_t_foo = wutils.wave_sub_list(wutils.wave_mat2list(x_1_bar_cap_t),wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(r1_bar_t), D1_t))
            r2_bar_t = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(r2_bar_t_foo,fac1))

            GAMMA_2_full = GAMMA_1_full*((1/D1_t) - 1)
            GAMMA_2_full = torch.clip(GAMMA_2_full, 1e-21, 1e21)
            
            
            # Denoiser Stage
            x_2_bar_cap_t, D2_t = ColoredDnCNN_GEC_denoiser_complex(r2_bar_t, 1/(GAMMA_2_full))
            D2_t = D2_t.reshape(1,3*level + 1)
            D2_t = torch.abs(D2_t)

            recovered_image_Denoiser = wutils.wave_inverse_mat(x_2_bar_cap_t,ifm,level)
            recovered_image_Denoiser_abs = transforms_new.complex_abs((wutils.wave_inverse_mat(x_2_bar_cap_t.clone(),ifm, level)).squeeze(0).permute(1,2,0))

            if use_DDVAMP_damping == True:
                if iter>0:
                    D2_t = (theta_damp*torch.sqrt(D2_t) + (1 - theta_damp)*torch.sqrt(D2_t_old))**2

            D2_t_old = D2_t.clone()
            fac11 = 1/(1 - D2_t)

            r1_bar_t_foo = wutils.wave_sub_list(wutils.wave_mat2list(x_2_bar_cap_t),wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(r2_bar_t), D2_t))
            r1_bar_t = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(r1_bar_t_foo,fac11))

            GAMMA_1_full = GAMMA_2_full*((1/D2_t) - 1)
            GAMMA_1_full = torch.clip(GAMMA_1_full, 1e-21, 1e21)

            if use_DDVAMP_damping == True:
                if iter>0:
                    r1_bar_t = wutils.get_wave_mat(wutils.wave_add_list(wutils.wave_scalar_mul_list(wutils.wave_mat2list(r1_bar_t),zeta_damp),wutils.wave_scalar_mul_list(wutils.wave_mat2list(r1_bar_t_old),(1 - zeta_damp))))
                    GAMMA_1_full = (1/(zeta_damp*torch.sqrt(1/GAMMA_1_full) + (1-zeta_damp)*torch.sqrt(1/GAMMA_1_full_old)))**2

            r1_bar_t_old = r1_bar_t.clone()
            GAMMA_1_full_old = GAMMA_1_full.clone()
            
            PSNR_list.append(gutil.calc_psnr((recovered_image_Denoiser_abs*metric_mask).cpu(), (GT_target*metric_mask).cpu(), max = (GT_target*metric_mask).max().cpu()))
                
    return recovered_image_Denoiser, recovered_image_LMMSE, PSNR_list
