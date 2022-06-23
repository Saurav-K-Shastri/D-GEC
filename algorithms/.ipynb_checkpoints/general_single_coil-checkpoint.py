import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch


from utils import general as gutil
from utils import transform as tutil
from utils import my_transforms as mutil
from utils import *

from fastMRI_utils import transforms_new

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

import utils.wave_torch_transforms as wutils

from numpy import linalg as LA

def gen_pdf(shape, sampling_rate, p=8, dist_type='l2', radius=0., ret_tensor=False):
    """Generate probability density function (PDF) for variable density undersampling masking in MRI simulation

    Args:
        shape: shape of image
        sampling_rate (float): ratio of sampled pixels to ground truth pixels (n/N)
        p (int): polynomial power
        dist_type (str): distance type - l1 or l2
        radius (float): radius of fully sampled center

    Returns:
        pdf (np.ndarray): the desired PDF (sampling probability map)

    Notes:
        This is the Python implementation of the genPDF function from the SparseMRI package.
        (http://people.eecs.berkeley.edu/~mlustig/Software.html). The sampling scheme is described
        in the paper M. Lustig, D.L Donoho and J.M Pauly “Sparse MRI: The Application of Compressed
        Sensing for Rapid MR Imaging” Magnetic Resonance in Medicine, 2007 Dec; 58(6):1182-1195.

    """
    C, H, W = shape

    num_samples = np.floor(sampling_rate * H * W)

    x, y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))
    if dist_type == 'l1':
        r = np.maximum(np.abs(x), np.abs(y))
    elif dist_type == 'l2':
        r = np.sqrt(x ** 2 + y ** 2)
        r /= np.max(np.abs(r))
    else:
        raise ValueError('genPDF: invalid dist_type')

    idx = np.where(r < radius)

    pdf = (np.ones_like(r) - r) ** p
    pdf[idx] = 1

    if np.floor(np.sum(pdf)) > num_samples:
        raise RuntimeError('genPDF: infeasible without undersampling dc, increase p')

    # Bisection
    minval = 0
    maxval = 1
    val = 0.5
    it = 0
    for _ in range(20):
        it += 1
        val = (minval + maxval) / 2
        pdf = (np.ones_like(r) - r) ** p + val * np.ones_like(r)
        pdf[np.where(pdf > 1)] = 1
        pdf[idx] = 1
        N = np.floor(np.sum(pdf))
        if N > num_samples:		# Infeasible
            maxval = val
        elif N < num_samples:	# Feasible, but not optimal
            minval = val
        elif N == num_samples:	# Optimal
            break
        else:
            raise RuntimeError('genPDF: error with calculation of N')

    if ret_tensor:
        return torch.from_numpy(pdf).to(dtype=torch.float32)
    else:
        return pdf
    
    

class B_op:
    
    def __init__(self,idx1_complement,idx2_complement,xfm,ifm, level):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.xfm = xfm
        self.ifm= ifm
        self.level = level 
        
    def A(self,X):
        X1 = (wutils.wave_inverse_mat(X,self.ifm, self.level)).permute(0,2,3,1)
        out = transforms_new.fft2c_new(X1)
        out[:,self.idx1_complement,self.idx2_complement,:] = 0
        return out.permute(0,3,1,2)

    def H(self,X):
        X = X.permute(0,2,3,1)
        X[:,self.idx1_complement,self.idx2_complement,:] = 0
        out1 = transforms_new.ifft2c_new(X)
        out = wutils.wave_forward_mat(out1.permute(0,3,1,2),self.xfm)
        return out
    
    
## Wavelet measurements

def get_single_coil_noisy_measurement(loadlist, image_number, snr, sampling_rate, device):

    target = gutil.read_image(loadlist[image_number]) 
    min_val = target.min()
    max_val = target.max()
    target = (target - min_val)/(max_val - min_val)
    target_np = target[0,:,:].numpy()
    target_complex = torch.zeros(2,target.shape[1],target.shape[2])
    target_complex[0,:,:] = target

    # Sampling Mask
    prob_map = gen_pdf(target.shape, sampling_rate)
    mask = np.random.binomial(1, prob_map)
    idx1_complement = np.where(mask == 0)[0]
    idx2_complement = np.where(mask == 0)[1]
    idx1 = np.where(mask == 1)[0]
    idx2 = np.where(mask == 1)[1]
    
    wavelet = 'haar'
    level = 4
    num_of_sbs = 3*level + 1

    xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)

    x0 = wutils.wave_forward_mat(target_complex.unsqueeze(0).to(device),xfm)
    x0 = x0.to(device)
    N = x0.shape[-1]**2 # Assuming square image
    n = x0.shape[-1]

    B_op_foo = B_op(idx1_complement,idx2_complement,xfm,ifm, level)

    y0 = B_op_foo.A(x0) # measurement; size 1x2xnxn
    M = len(idx1)
    
    yvar = torch.sum(transforms_new.complex_abs(y0.permute(0,2,3,1))**2)/M
    wvar = yvar*torch.pow(torch.tensor(10), -0.1*snr)
    yshape = y0.shape
    y = wutils.add_noise_to_complex_measurements_no_verbose(y0,wvar,idx1_complement,idx2_complement,device,is_complex = True) # noisy measurement
    y_np = transforms_new.complex_abs((y.clone()).permute(0,2,3,1).squeeze(0).cpu()).numpy()
    
    return y, target, mask, prob_map, wvar, M, N