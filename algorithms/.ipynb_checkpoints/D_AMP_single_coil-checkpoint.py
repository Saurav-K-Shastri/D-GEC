import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch

from algorithm import dvdamp
from algorithm import simulation as sim
from algorithm import denoiser as den
from algorithm import heatmap
from utils import general as gutil
from utils import plot as putil
from utils import transform as tutil
from utils import my_transforms as mutil
from utils.general import load_checkpoint, save_image
from utils import GEC_pytorch_denoiser_util as dutils
from utils import *


from algorithm import csalgo

from fastMRI_utils import transforms_new

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

import utils.wave_torch_transforms as wutils

from numpy import linalg as LA

from models.model import Colored_DnCNN 



class A_op_D_AMP:
    
    def __init__(self,mask):
        self.mask = mask
        
    def A(self,X):
        out1 = tutil.fftnc(X.cpu().numpy(), ret_tensor=True)
        out = self.mask*out1
        return out

    def H(self,X):
        
        X1 = self.mask*X
        out = tutil.ifftnc(X.cpu().numpy(), ret_tensor=True)
        return out




def D_AMP(y, mask, num_of_D_AMP_iterations, trained_DnCNN_white):

    complex_weight = 0
    device = y.device

    y_dummy = (y.clone()).permute(0,2,3,1).squeeze(0).cpu().numpy()
    y_np = y_dummy[:,:,0] + 1j*y_dummy[:,:,1]
    
    y_new = torch.from_numpy(y_np).to(dtype=torch.complex64)
    
    
    model_denoiser = load_model(trained_DnCNN_white)    
    model_denoiser.to(device=device)
    model_denoiser.eval()
    
        
    idx1_complement = np.where(mask == 0)[0]
    idx2_complement = np.where(mask == 0)[1]
    idx1 = np.where(mask == 1)[0]
    idx2 = np.where(mask == 1)[1]

    n = mask.shape[-1]**2 # Assuming square image
    m = len(idx1)
    
    cs_algo = csalgo.DAMP_complex([1,mask.shape[0],mask.shape[1]], model_denoiser, num_of_D_AMP_iterations, image=None)
    
    mask = torch.from_numpy(mask)
    
    mri = A_op_D_AMP(mask)
        
    recon_image, r_t, std_est = cs_algo(y_new, mri.A, mri.H,m,n,device,complex_weight)
    
    recon_damp = torch.zeros_like(y)
    recon_damp[:,0,:,:] = torch.real(recon_image.to(device))
    recon_damp[:,1,:,:] = torch.imag(recon_image.to(device))
    
    return recon_damp


def D_AMP2(y, mask, num_of_D_AMP_iterations, trained_DnCNN_white, GT_target):

    complex_weight = 0
    device = y.device

    y_dummy = (y.clone()).permute(0,2,3,1).squeeze(0).cpu().numpy()
    y_np = y_dummy[:,:,0] + 1j*y_dummy[:,:,1]
    
    y_new = torch.from_numpy(y_np).to(dtype=torch.complex64)
    
    
    model_denoiser = load_model(trained_DnCNN_white)    
    model_denoiser.to(device=device)
    model_denoiser.eval()
    
        
    idx1_complement = np.where(mask == 0)[0]
    idx2_complement = np.where(mask == 0)[1]
    idx1 = np.where(mask == 1)[0]
    idx2 = np.where(mask == 1)[1]

    n = mask.shape[-1]**2 # Assuming square image
    m = len(idx1)
    
    cs_algo = csalgo.DAMP_complex2([1,mask.shape[0],mask.shape[1]], model_denoiser, num_of_D_AMP_iterations, image=None)
    
    mask = torch.from_numpy(mask)
    
    mri = A_op_D_AMP(mask)
        
    recon_image, recon_image_mat, r_t, std_est = cs_algo(y_new, mri.A, mri.H,m,n,device,complex_weight)
    
    recon_damp = torch.zeros_like(y)
    recon_damp[:,0,:,:] = torch.real(recon_image.to(device))
    recon_damp[:,1,:,:] = torch.imag(recon_image.to(device))

    PSNR_list_DAMP = []
    
    for i in range(num_of_D_AMP_iterations):
        
            recon_damp_dummy = torch.zeros_like(y)
            recon_damp_dummy[:,0,:,:] = torch.real(recon_image_mat[i,:,:].to(device))
            recon_damp_dummy[:,1,:,:] = torch.imag(recon_image_mat[i,:,:].to(device))
            
            PSNR_list_DAMP.append(gutil.calc_psnr((recon_damp_dummy[0,0,:,:]).cpu(), (GT_target[0,0,:,:]).cpu(), max = (GT_target[0,0,:,:]).max().cpu()))
        
    return recon_damp, PSNR_list_DAMP