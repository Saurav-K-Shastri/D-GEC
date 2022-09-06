import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from algorithm.denoiser import ColoredDnCNN
from fastMRI_utils import transforms_new

from utils import *

from utils.general import load_checkpoint, save_image
import time


import torch
from utils import my_transforms as mutil
from fastMRI_utils import transforms_new
import utils.wave_torch_transforms as wutils

import matplotlib.pyplot as plt

class B_op:
    
    def __init__(self,idx1_complement,idx2_complement,xfm,ifm, level):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.xfm = xfm
        self.ifm = ifm
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
    
class B_op_DC:
    
    def __init__(self,idx1_complement,idx2_complement,xfm,ifm,level, Dmh):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.xfm = xfm
        self.ifm= ifm
        self.level = level 
        self.Dmh = Dmh
        
    def A(self,X):
        X1 = (wutils.wave_inverse_mat(X,self.ifm, self.level)).permute(0,2,3,1)
        out = transforms_new.fft2c_new(X1)
        out[:,self.idx1_complement,self.idx2_complement,:] = 0
        return (self.Dmh)*(out.permute(0,3,1,2))

    def H(self,X):
        X = (self.Dmh*X).permute(0,2,3,1)
        X[:,self.idx1_complement,self.idx2_complement,:] = 0
        out1 = transforms_new.ifft2c_new(X)
        out = wutils.wave_forward_mat(out1.permute(0,3,1,2),self.xfm)
        return out
    
    
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
    
    
    
class INP_op:

    def __init__(self,level,my_scalar_vec):
        self.level = level 
        self.my_scalar_vec = my_scalar_vec
        
    def forward(self, y, r):
        
        out1 = y
        
        out2 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(r,level = self.level),self.my_scalar_vec))
        
        return torch.cat((out1,out2),dim = 2)
    
    
class CG_INP_op:

    def __init__(self,B_op,level,my_scalar_vec):
        self.B_op = B_op
        self.level = level 
        self.my_scalar_vec = my_scalar_vec
        
    def forward(self, y, r):
        
        out1 = self.B_op.H(y)
        
        out2 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(r,level = self.level),self.my_scalar_vec))
        
        out = out1+out2
        
        return out
    
class CG_INP_precond_op:

    def __init__(self,B_op,level,my_scalar_vec):
        self.B_op = B_op
        self.level = level 
        self.my_scalar_vec = my_scalar_vec
        
    def forward(self, y, r):
        
        out1 = self.B_op.H(y)
        out11 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(out1,level = self.level),self.my_scalar_vec))
        
        out2 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(r,level = self.level),torch.pow(self.my_scalar_vec,-1)))
        
        out = out11+out2
        
        return out
    
    
class CG_op:

    def __init__(self,B_op,level,my_scalar_vec):
        self.B_op = B_op
        self.level = level
        self.my_scalar_vec = my_scalar_vec
        
    def forward(self, z):
        
        out1 = self.B_op.H(self.B_op.A(z))
        
        out2 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(z,level = self.level),self.my_scalar_vec))
        
        out = out1 + out2
        
        return out 
    
class CG_precond_op:

    def __init__(self,B_op,level,my_scalar_vec):
        self.B_op = B_op
        self.level = level
        self.my_scalar_vec = my_scalar_vec
        
    def forward(self, z):
        
        out1 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(z,level = self.level),self.my_scalar_vec))
        
        out2 = self.B_op.H(self.B_op.A(out1))
        
        out3 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(out2,level = self.level),self.my_scalar_vec))
        
        out = out3 + z
        
        return out 
    

    
    
    
def CG_method(b,OP_A,x,max_iter,eps_lim):
    r = b - OP_A.forward(x)
    p = r.clone()
    
    rsold = (torch.real(torch.dot(torch.conj(torch.view_as_complex(r.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(r.permute(0,2,3,1).contiguous()).reshape(-1))))
    for i in range(max_iter):
        Ap = OP_A.forward(p)

        alpha = rsold/(torch.real(torch.dot(torch.conj(torch.view_as_complex(p.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(Ap.permute(0,2,3,1).contiguous()).reshape(-1))))

#         print(i)
        x = x + alpha*p
        r = r - alpha*Ap
        
        rsnew = (torch.real(torch.dot(torch.conj(torch.view_as_complex(r.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(r.permute(0,2,3,1).contiguous()).reshape(-1))))
        
        
        if torch.sqrt(rsnew) < eps_lim:
            break
        
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x,i,torch.sqrt(rsnew)


def CG_method_subbands(b,OP_A,x,max_iter): # works for image+subband batches as well
    
    r = b - OP_A.forward(x)
    p = r.clone()


    
    rsold = transforms_new.real_part_of_aHb(r.permute(0,2,3,1),r.permute(0,2,3,1))
    
    
    
    for i in range(max_iter):
        Ap = OP_A.forward(p)

        alpha = rsold/transforms_new.real_part_of_aHb(p.permute(0,2,3,1),Ap.permute(0,2,3,1))

        x = x + (alpha*p.permute(1,2,3,0)).permute(3,0,1,2)
        r = r - (alpha*Ap.permute(1,2,3,0)).permute(3,0,1,2)
        
        rsnew = transforms_new.real_part_of_aHb(r.permute(0,2,3,1),r.permute(0,2,3,1))
        

    
        p = r + ((rsnew/rsold)*p.permute(1,2,3,0)).permute(3,0,1,2)
        rsold = rsnew

    
    return x,i,torch.sqrt(rsnew)



def calc_batch_MC_divergence_complex_with_warm_strt_2(denoiser, wavelet_mat, variances, x_init_with_MC, level,subband_sizes,p1m1_mask,changeFactor):

    """This is for processing multiple subband batches only
    """
    device = wavelet_mat.device

    
    next_x_warm_MC = torch.zeros_like(x_init_with_MC)
    
    
    alpha = torch.zeros(3*level + 1, device = device)

    
    eta1 = wutils.get_mean_in_each_subband(wavelet_mat,p1m1_mask,subband_sizes)
        
    eta2 = torch.sqrt(variances)
    

    eta_subband = changeFactor*torch.min(eta1,eta2)
    
    eps = torch.tensor(2.22e-16, device = device)
    
    eta_subband = eta_subband + eps 
    
    batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch_subband(wavelet_mat,level,eta_subband,p1m1_mask)

    
    denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances, x_init_with_MC)

    next_x_warm_with_MC = (denoised_jittered_with_denoised).clone()
    denoised = denoised_jittered_with_denoised[0,:,:,:].unsqueeze(0)
    denoised_jittered = denoised_jittered_with_denoised[1:,:,:,:]

    alpha = (1. / subband_sizes)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),(((denoised_jittered - denoised).permute(1,2,3,0)/eta_subband).permute(3,0,1,2)).permute(0,2,3,1))
    
    return denoised, alpha, next_x_warm_with_MC




def calc_batch_MC_divergence_true_complex_2(denoiser, wavelet_mat, variances, level,subband_sizes,p1m1_mask, ifm, scale_percentile,changeFactor):

    """This is for processing multiple subband batches only
    """
    # Find the scaling factor
    
    
    noisy_image = wutils.wave_inverse_mat(wavelet_mat, ifm, level)
    sorted_image_vec = transforms_new.complex_abs(noisy_image.squeeze(0).permute(1,2,0)).reshape((-1,)).sort()
    scale = sorted_image_vec.values[int(len(sorted_image_vec.values) * scale_percentile/100)].item()
    
    device = wavelet_mat.device
           
    alpha = torch.zeros(3*level + 1, device = device)

#     eta1 = wutils.get_max_in_each_subband(wavelet_mat,p1m1_mask)/10
    eta1 = wutils.get_mean_in_each_subband(wavelet_mat,p1m1_mask,subband_sizes)
        
    eta2 = torch.sqrt(variances)
    
#     print('LMMSE Stage delta selected pos --> eta2 (var), neg --> eta 1 (abs (r))')
#     print(torch.sign(eta1-eta2))
    
    eta_subband = changeFactor*torch.min(eta1,eta2)
#     eta_subband = changeFactor*torch.min(eta2)
        
    eps = torch.tensor(2.22e-16, device = device)
    
    eta_subband = eta_subband + eps 
    
    batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch_subband(wavelet_mat,level,eta_subband,p1m1_mask)
    
    denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances, scale)

    denoised = denoised_jittered_with_denoised[0,:,:,:].unsqueeze(0)
    denoised_jittered = denoised_jittered_with_denoised[1:,:,:,:]
    
    alpha = (1. / subband_sizes)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),(((denoised_jittered - denoised).permute(1,2,3,0)/eta_subband).permute(3,0,1,2)).permute(0,2,3,1))
    
    return denoised, alpha

 

class LMMSE_batch_CG_GEC_with_div_and_warm_strt_multi_coil:
    """Wrapper of LMMSE for using with GEC which used CG. This code uses warm start. This is for multicoil"""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w, xfm,ifm, p1m1_mask, subband_sizes, sense_map, level = 4,LMMSE_inner_iter_lim = 100, beta_tune_LMMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4), changeFactor = 0.1):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.level = level
        self.LMMSE_inner_iter_lim = LMMSE_inner_iter_lim
        self.beta_tune_LMMSE = beta_tune_LMMSE
        self.eps_lim = eps_lim
        self.subband_sizes = subband_sizes
        self.p1m1_mask = p1m1_mask
        self.xfm = xfm
        self.ifm = ifm
        self.sense_map = sense_map
        self.changeFactor = changeFactor
        self.sigma_w = sigma_w/torch.sqrt(self.beta_tune_LMMSE)
        
    def __call__(self, wavelet_mat, variances, x_init_with_MC, calc_divergence=True):

#         denoised, alpha, next_x_warm_with_MC = calc_batch_MC_divergence_complex_with_warm_strt(self._denoise, wavelet_mat, variances, x_init_with_MC, self.level,self.subband_sizes,self.p1m1_mask)
        denoised, alpha, next_x_warm_with_MC = calc_batch_MC_divergence_complex_with_warm_strt_2(self._denoise, wavelet_mat, variances, x_init_with_MC, self.level,self.subband_sizes,self.p1m1_mask, self.changeFactor)
        
        return denoised, alpha, next_x_warm_with_MC


    def _denoise(self, wavelet_mat, variances, x_init):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = (self.sigma_w**2)*(GAMMA_1_full)
        
        B_op_foo = B_op_multi(self.idx1_complement,self.idx2_complement, self.xfm, self.ifm, self.level, self.sense_map)       
        A_bar = CG_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_op_foo = CG_INP_op(B_op_foo,self.level,my_scalar_vec) 
        b_bar = CG_INP_op_foo.forward(self.y,wavelet_mat)

        denoised_wavelet,stop_iter,rtr_end = CG_method_subbands(b_bar,A_bar,x_init,self.LMMSE_inner_iter_lim) 

        return denoised_wavelet
    

    
class DnCNN_cpc_VDAMP_true_complex_batch:
    """
    handles true complex denoising; used mainly for fastMRI data; 
    """
    def __init__(self, modeldir, std_ranges, xfm,ifm, p1m1_mask, subband_sizes,  channels=2, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = torch.tensor(1), device=torch.device('cpu'),
                std_pool_func=torch.mean, verbose=False, level = 4, scale_percentile = 98, changeFactor = 0.1):

        self.channels = channels
        self.std_ranges = std_ranges
        self.wavetype = wavetype
        self.device = device
        self.models = self._load_models(modeldir)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.beta_tune = beta_tune
        self.level = level
        self.subband_sizes = subband_sizes
        self.p1m1_mask = p1m1_mask
        self.xfm = xfm
        self.ifm = ifm
        self.scale_percentile = scale_percentile
        self.changeFactor = changeFactor
        
    def __call__(self, wavelet_mat, variances, calc_divergence=True):

        variances *= self.beta_tune
        
        denoised, alpha = calc_batch_MC_divergence_true_complex_2(self._denoise, wavelet_mat, variances, self.level,self.subband_sizes,self.p1m1_mask, self.ifm, self.scale_percentile,self.changeFactor)
        
        return denoised, alpha
    
    
    @torch.no_grad()
    def _denoise(self, wavelet_mat, true_variances, scale): 
        
        #Scale Variance
        variances = (true_variances.clone())/(scale**2)
        
        # Select the model to use
        stds = torch.sqrt(variances)
        std_pooled = self.std_pool_func(torch.sqrt(variances))
        select = torch.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('DnCNN_cpc_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = wutils.wave_inverse_mat(wavelet_mat, self.ifm, self.level)
        
        # Scale Image
        noisy_image_scaled = (noisy_image.clone())/scale

        noise_realization_mat = wutils.wave_inverse_list(wutils.add_noise_subbandwise_list(wutils.wave_mat2list(torch.zeros_like(noisy_image_scaled), self.level),stds),self.ifm)
        
        model_inp = torch.cat(((noisy_image_scaled).clone(),noise_realization_mat[:,0,:,:].unsqueeze(1)), dim = 1)

#             ttt = time.time()
        denoised_image_scaled = self.models[select](model_inp)
#             print("elapsed: ", time.time() - ttt)

        
        # Un-Scale Image
        
        denoised_image = (denoised_image_scaled.clone())*scale
        
        denoised_wavelet_mat = wutils.wave_forward_mat(denoised_image,self.xfm)

        return denoised_wavelet_mat

    def _load_models(self, modeldirs):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = load_model(modeldir)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    

def correct_GAMMA_using_EM_with_mask_LMMSE(denoiser,noisy_wavelet_mat, variances, warm_start_mat, num_em_iter, level, wave_mask_list):

    """Correct the variances entering the denoiser using EM iterations

    """
    device = noisy_wavelet_mat.device
#     print('running EM correction for LMMSE')
    
    for i in range(num_em_iter):
                
        denoised, alpha, warm_start_mat = denoiser(noisy_wavelet_mat, variances,warm_start_mat,True)

        variances = torch.as_tensor(wutils.find_subband_wise_MSE_list_with_wave_mask(wutils.wave_mat2list(denoised,level),wutils.wave_mat2list(noisy_wavelet_mat,level), wave_mask_list), device = device)+ variances*torch.as_tensor(alpha, device = device)
        
    return variances



def correct_GAMMA_using_EM_with_mask(denoiser,noisy_wavelet_mat, variances, num_em_iter, level, wave_mask_list):

    """Correct the variances entering the denoiser using EM iterations

    """
    device = noisy_wavelet_mat.device
#     print('running EM correction')
    
    for i in range(num_em_iter):
                
        denoised, alpha = denoiser(noisy_wavelet_mat, variances)
        
        variances = torch.as_tensor(wutils.find_subband_wise_MSE_list_with_wave_mask(wutils.wave_mat2list(denoised,level),wutils.wave_mat2list(noisy_wavelet_mat,level), wave_mask_list), device = device)+ variances*torch.as_tensor(alpha, device = device)
        
    return variances

