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
from fastMRI_utils.utils_fastMRI import tensor_to_complex_np
from fastMRI_utils.mri_data_new import SelectiveSliceData_Train
from fastMRI_utils.mri_data_new import SelectiveSliceData_Val

from fastMRI_utils.mri_data_new import SelectiveSliceData_Train_Brain
from fastMRI_utils.mri_data_new import SelectiveSliceData_Val_Brain


from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

import scipy.misc

import cv2
import sigpy as sp
import sigpy.mri as mr

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
    

class A_op_multi_fully_sampled:
    
    def __init__(self,sens_map):
        self.sens_map = sens_map
        self.nc = sens_map.shape[0]
        
    def A(self,X):
        X = (X).permute(0,2,3,1)
        X_sens = transforms_new.complex_mult(X.unsqueeze(1).repeat(1,self.nc,1,1,1), self.sens_map.unsqueeze(0))
        out_foo = transforms_new.fft2c_new(X_sens)
        out = torch.cat((out_foo[:,:,:,:,0], out_foo[:,:,:,:,1]), dim = 1)
        return out

    def H(self,X):
        X = X.permute(0,2,3,1)
        X_new = torch.stack([X[:,:,:,0:self.nc],X[:,:,:,self.nc:]], dim = -1).permute(0,3,1,2,4)
        out_sens = transforms_new.ifft2c_new(X_new)
        out_foo = torch.sum(transforms_new.complex_mult(out_sens, transforms_new.complex_conj(self.sens_map.unsqueeze(0))),dim = 1)
        out = out_foo.permute(0,3,1,2)
        
        return out
    
    
class Init_Arg:
    def __init__(self):
        self.seed=42
        self.resolution=368
        self.challenge='multicoil'
        self.sample_rate=1.  
        self.output_path = None
        self.accel=False
        self.use_mid_slices=True    
        self.scanner_strength=3
        self.scanner_mode = 'PD'
        self.num_of_top_slices = 8
        self.num_of_mid_slices = 8
        self.image_size = 320
        self.scaling_mode = 'percentile'
#         self.scaling_mode = 'absolute_max'
#         self.scaling_mode = 'constant'
        self.percentile_scale = 98
#         self.constant_scale = 0.0005345118
        self.constant_scale = 0.0012
    
    
def Rss(x):
    y = np.expand_dims(np.sum(np.abs(x)**2,axis = -1)**0.5,axis = 2)
    return y

def ImageCropandKspaceCompression(x,image_size):
#     print(x.shape)
#     plt.imshow(np.abs(x[:,:,0]), origin='lower', cmap='gray')
#     plt.show()
        
    w_from = (x.shape[0] - image_size) // 2  # crop images into 320x320
    h_from = (x.shape[1] - image_size) // 2
    w_to = w_from + image_size
    h_to = h_from + image_size
    cropped_x = x[w_from:w_to, h_from:h_to,:]
    
#     print('cropped_x shape: ',cropped_x.shape)
    if cropped_x.shape[-1] >= 8:
        x_tocompression = cropped_x.reshape(image_size**2,cropped_x.shape[-1])
        U,S,Vh = np.linalg.svd(x_tocompression,full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:,0:8].reshape(image_size,image_size,8)
    else:
        coil_compressed_x = cropped_x
        
    return coil_compressed_x


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, use_seed=False):
        
        self.use_seed = use_seed
        self.args = args
        self.mask = None
        self.resolution = args.resolution
        self.scaling_mode = args.scaling_mode
        self.percentile_scale = args.percentile_scale
        self.constant_scale = args.constant_scale
        
        
    def __call__(self, kspace):

        fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
        ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

        numcoil = 8

        kspace = kspace.transpose(1,2,0) 

        x = ifft(kspace, (0,1)) #(768, 396, 16)

        coil_compressed_x = ImageCropandKspaceCompression(x,self.resolution) #(384, 384, 8)
        
        my_taper_mask = (torch.load('my_taper_mask.pt')).numpy()        
        for  i in range(8):
            coil_compressed_x[:,:,i] = np.multiply(coil_compressed_x[:,:,i],my_taper_mask)
            
        RSS_x = np.squeeze(Rss(coil_compressed_x))# (384, 384)

        kspace = fft(coil_compressed_x, (1,0)) #(384, 384, 8)

        kspace = transforms_new.to_tensor(kspace)

        kspace = kspace.permute(2,0,1,3)
 
        kspace_np = tensor_to_complex_np(kspace)

        ESPIRiT_tresh = 0.02
#         ESPIRiT_crop = 0.96
        ESPIRiT_crop = 0.95

        ESPIRiT_width_full = 24
        ESPIRiT_width_mask = 24
        device=sp.Device(-1)
        
        sens_maps = mr.app.EspiritCalib(kspace_np,calib_width= ESPIRiT_width_mask,thresh=ESPIRiT_tresh, kernel_width=6, crop=ESPIRiT_crop,device=device,show_pbar=False).run()
        
        sens_maps = sp.to_device(sens_maps, -1)
        
        sens_map_foo = np.zeros((self.resolution,self.resolution,8)).astype(np.complex128)
        
        for  i in range(8):
            sens_map_foo[:,:,i] = sens_maps[i,:,:]

        lsq_gt = np.sum(sens_map_foo.conj()*coil_compressed_x , axis = -1)   
        
        image = transforms_new.to_tensor(lsq_gt)
        
        if self.scaling_mode == 'percentile':
            sorted_image_vec = transforms_new.complex_abs(image).reshape((-1,)).sort()
            scale = sorted_image_vec.values[int(len(sorted_image_vec.values) * self.percentile_scale/100)].item()
        elif self.scaling_mode == 'absolute_max':
            scale = transforms_new.complex_abs(image).max()
        else:
            scale = self.constant_scale #  number obtained by taking the mean value of the max values of the training images (check Single_coil_Knee_Data_Access.ipynb)
        
#         scale = 0.0012 # constant scale, This is the scaling Ted used in his codes. 
        
#         image_abs = transforms_new.complex_abs(image)
        
        image = image/scale
        image = image.permute(2,0,1)
        
        return image, kspace, transforms_new.to_tensor(sens_map_foo).permute(2,0,1,3)
    
    

def create_data_loader(args):

    dev_data = SelectiveSliceData_Val(
        root=args.data_path_val,
        transform=DataTransform(args),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True, # Set true. It actually uses mid slices
        number_of_top_slices=args.number_of_mid_slices, #Although it says top slices, the code uses mid 8 slices
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )
        
    return dev_data


def create_data_loader_knee(args):

    dev_data = SelectiveSliceData_Val(
        root=args.data_path_val,
        transform=DataTransform(args),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True, # Set true. It actually uses mid slices
        number_of_top_slices=args.number_of_mid_slices, #Although it says top slices, the code uses mid 8 slices
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )
        
    return dev_data

def create_data_loader_brain(args):

    dev_data = SelectiveSliceData_Val_Brain(
        root=args.data_path_val,
        transform=DataTransform(args),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.number_of_mid_slices, 
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )
        
    return dev_data



def get_multi_coil_data(data_path,number_of_mid_slices = 4, dataset='knee'):
    
    args = Init_Arg()
    args.data_path_val = data_path
    args.number_of_mid_slices = 4
    
    if dataset == 'knee':
        dev_data = create_data_loader_knee(args) 
    else:
        dev_data = create_data_loader_brain(args) 
    
    return dev_data


def get_multi_coil_noisy_measurement_and_sens_maps(dev_data, image_number, snr, sampling_rate, device):
    # Note: The output Sens Maps are estimated using noisy undersampled measurements

    nc = 8 # number of coils
    resolution = 368
    
    wavelet = 'haar'
    level = 4
    num_of_sbs = 3*level + 1

    xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)
    
    target_complex, full_kspace, GT_sens_maps = dev_data[image_number]

    target_complex = target_complex.type('torch.FloatTensor')
    full_kspace = full_kspace.type('torch.FloatTensor')
    GT_sens_maps = GT_sens_maps.type('torch.FloatTensor')

    # target_complex is scalled target.  To get the actual target, we use fully sampled k-space data
    A_op_foo_full = A_op_multi_fully_sampled(GT_sens_maps.to(device))
    y_full = (torch.cat((full_kspace[:,:,:,0], full_kspace[:,:,:,1]), dim = 0).unsqueeze(0)).to(device)
    GT_target_complex = (A_op_foo_full.H(y_full))
    GT_target_abs = transforms_new.complex_abs(GT_target_complex.squeeze(0).permute(1,2,0))
    
    # Sampling Mask
    prob_map = gen_pdf([1, resolution, resolution], 1/((1/sampling_rate)-0.001)) # 0.001 accomodates the calibration region addition in the next step
    prob_map[int((resolution/2)-12):int((resolution/2)+12),int((resolution/2)-12):int((resolution/2)+12)] = 1 # for ESPIRiT fully sampled 24x24 calibration region. 
    
    mask = np.random.binomial(1, prob_map)
    idx1_complement = np.where(mask == 0)[0]
    idx2_complement = np.where(mask == 0)[1]
    idx1 = np.where(mask == 1)[0]
    idx2 = np.where(mask == 1)[1]

    x0 = wutils.wave_forward_mat(target_complex.unsqueeze(0).to(device),xfm)
    x0 = x0.to(device)
    N = x0.shape[-1]**2 # Assuming square image
    n = x0.shape[-1]
    M = resolution*resolution - len(idx1_complement)
    
    masked_kspace = full_kspace.clone()
    masked_kspace[:,idx1_complement,idx2_complement,:] = 0

    masked_kspace = masked_kspace.to(device)

    y0 = torch.cat((masked_kspace[:,:,:,0], masked_kspace[:,:,:,1]), dim = 0).unsqueeze(0)
    
    y_complex = torch.stack([y0.permute(0,2,3,1)[:,:,:,0:nc],y0.permute(0,2,3,1)[:,:,:,nc:]], dim = -1).permute(0,3,1,2,4)

    
    yvar = torch.sum(transforms_new.complex_abs(y_complex)**2)/(M*nc)
    wvar = yvar*torch.pow(torch.tensor(10), -0.1*snr)
    yshape = y0.shape
    y = wutils.add_noise_to_complex_measurements_no_verbose(y0,wvar,idx1_complement,idx2_complement,device,is_complex = True) # noisy measurement
    # y_np = transforms_new.complex_abs((y.clone()).permute(0,2,3,1).squeeze(0).cpu()).numpy()

    
    ## Sensitivity Map Computation
    
    y_foo = y.clone()
    masked_kspace_noisy = torch.stack([y_foo.permute(0,2,3,1)[:,:,:,0:nc],y_foo.permute(0,2,3,1)[:,:,:,nc:]], dim = -1).permute(0,3,1,2,4).squeeze(0).cpu()
    masked_kspace_np = tensor_to_complex_np(masked_kspace_noisy)
    sens_maps = mr.app.EspiritCalib(masked_kspace_np,calib_width= 24,thresh=0.02, kernel_width=6, crop=0.95,device=sp.Device(-1),show_pbar=False).run()
    sens_maps = sp.to_device(sens_maps, -1)
    sens_map_foo = np.zeros((n,n,8)).astype(np.complex128)
    for  i in range(8):
        sens_map_foo[:,:,i] = sens_maps[i,:,:]
    sens_maps_new = transforms_new.to_tensor(sens_map_foo).permute(2,0,1,3) 
    sens_maps_new = sens_maps_new.type('torch.FloatTensor')
    sens_maps_new = sens_maps_new.to(device)

    ###########
    
    
    dum = torch.ones(1,8,368,368,2,device = device)
    out_dum = transforms_new.complex_abs(torch.sum(transforms_new.complex_mult(dum, transforms_new.complex_conj(sens_maps_new.unsqueeze(0))),dim = 1)[0,:,:])

    metric_mask = torch.ones(out_dum.shape) - 1*(out_dum.cpu()==0)

    
    return y, GT_target_complex, sens_maps_new, mask, prob_map, wvar, M, N, metric_mask, GT_target_abs