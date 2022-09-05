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
    
    

# class CG_op:

#     def __init__(self,B_op,level,my_scalar_vec):
#         self.B_op = B_op
#         self.level = level
#         self.my_scalar_vec = my_scalar_vec
        
#     def forward(self, z):
        
#         out1 = self.B_op.A(z)
        
#         out2 = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(z,level = self.level),self.my_scalar_vec))
        
#         return torch.cat((out1,out2),dim = 2)
    
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
        
#         print(torch.sqrt(rsnew))
        
        if torch.sqrt(rsnew) < eps_lim:
            break
        
        p = r + (rsnew/rsold)*p
        rsold = rsnew
#     print(torch.sqrt(rsnew))
    return x,i,torch.sqrt(rsnew)


def CG_method_subbands(b,OP_A,x,max_iter): # works for image+subband batches as well
    
    r = b - OP_A.forward(x)
    p = r.clone()

#     rsnorm = torch.zeros(14,max_iter+1)
    
#     rsnorm[:,0] = (torch.sum((b - OP_A.forward(x))**2,[1,2,3]).cpu())
    
    rsold = transforms_new.real_part_of_aHb(r.permute(0,2,3,1),r.permute(0,2,3,1))
    
    
    
    for i in range(max_iter):
        Ap = OP_A.forward(p)

        alpha = rsold/transforms_new.real_part_of_aHb(p.permute(0,2,3,1),Ap.permute(0,2,3,1))
        
#         print(alpha)
        x = x + (alpha*p.permute(1,2,3,0)).permute(3,0,1,2)
        r = r - (alpha*Ap.permute(1,2,3,0)).permute(3,0,1,2)
        
        rsnew = transforms_new.real_part_of_aHb(r.permute(0,2,3,1),r.permute(0,2,3,1))
        
#         print(torch.sqrt(rsnew))
        
#         if torch.sqrt(rsnew) < eps_lim:
#             break
#         rsnorm[:,i+1] = (torch.sum((b - OP_A.forward(x))**2,[1,2,3]).cpu())
    
        p = r + ((rsnew/rsold)*p.permute(1,2,3,0)).permute(3,0,1,2)
        rsold = rsnew
#     print(torch.sqrt(rsnew))
    
#     plt.plot(rsnorm[0,-3:-1],label = "0")
#     plt.plot(rsnorm[1,-3:-1],label = "1")
#     plt.plot(rsnorm[2,-3:-1],label = "2")
#     plt.plot(rsnorm[3,-3:-1],label = "3")
#     plt.plot(rsnorm[4,-3:-1],label = "4")
#     plt.plot(rsnorm[5,-3:-1],label = "5")
#     plt.plot(rsnorm[6,-3:-1],label = "6")
#     plt.plot(rsnorm[7,-3:-1],label = "7")
#     plt.plot(rsnorm[8,-3:-1],label = "8")
#     plt.plot(rsnorm[9,-3:-1],label = "9")
#     plt.plot(rsnorm[10,-3:-1],label = "10")
#     plt.plot(rsnorm[11,-3:-1],label = "11")
#     plt.plot(rsnorm[12,-3:-1],label = "12")
#     plt.plot(rsnorm[13,-3:-1],label = "13")
#     plt.legend()
#     plt.show()
    
    return x,i,torch.sqrt(rsnew)


# def CG_method_subbands_batch(b,OP_A_batch,x,max_iter):
    
#     r = b - OP_A_batch.forward(x)
#     p = r.clone()
    
#     rsold = transforms_new.real_part_of_aHb(r.permute(0,1,3,4,2),r.permute(0,1,3,4,2))
    
    
#     for i in range(max_iter):
#         Ap = OP_A_batch.forward(p)

#         alpha = rsold/transforms_new.real_part_of_aHb(p.permute(0,1,3,4,2),Ap.permute(0,1,3,4,2))
        
# #         print(i)
#         x = x + (alpha*p.permute(2,3,4,0,1)).permute(3,4,0,1,2)
#         r = r - (alpha*Ap.permute(2,3,4,0,1)).permute(3,4,0,1,2)
        
#         rsnew = transforms_new.real_part_of_aHb(r.permute(0,1,3,4,2),r.permute(0,1,3,4,2))
        
# #         print(torch.sqrt(rsnew))
        
# #         if torch.sqrt(rsnew) < eps_lim:
# #             break
        
#         p = r + ((rsnew/rsold)*p.permute(2,3,4,0,1)).permute(3,4,0,1,2)
#         rsold = rsnew
# #     print(torch.sqrt(rsnew))
#     return x,i,torch.sqrt(rsnew)







def calc_MC_divergence_complex(denoiser, denoised, wavelet_mat, variances,level):

    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach.
    """
    device = wavelet_mat.device
    
    Yl,Yh = wutils.wave_mat2list(wavelet_mat,level)
    
    alpha = [None] * (3*level + 1)
    wavelet_mat_jittered = wavelet_mat.clone()
    eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances))
    

        
    eta = torch.max(eta1,eta2)
    eps = torch.tensor(2.22e-16, device = device)
    eta = eta + eps 
    
    noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),0)

    
    wavelet_mat_jittered += eta * noise_vec
    
    denoised_jittered = denoiser(wavelet_mat_jittered, variances)
    
    alpha[0] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(((denoised_jittered - denoised)/eta).permute(0,2,3,1).contiguous()).reshape(-1))))

    

    count = 1
    for s in range(level):
        index = level - 1 - s
        for b in range(3):
            
            wavelet_mat_jittered = wavelet_mat.clone()
            eta1 = torch.max(transforms_new.complex_abs(Yh[index][:,:,b,:,:].permute(0,2,3,1)))/1000.

                
            eta = torch.max(eta1,eta2)
            eps = torch.tensor(2.22e-16, device = device)
            eta = eta + eps
            
        
        
            noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),count)

            wavelet_mat_jittered += eta * noise_vec

            denoised_jittered = denoiser(wavelet_mat_jittered, variances)
            
            alpha[count] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(((denoised_jittered - denoised)/eta).permute(0,2,3,1).contiguous()).reshape(-1))))

            count = count + 1 
                    
    return torch.tensor(alpha, device = device)



def calc_MC_divergence_complex_with_warm_strt(denoiser, denoised, wavelet_mat, variances, x_init_MC, level):

    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach. This code uses warm start
    """
    device = wavelet_mat.device
    
    next_x_warm_MC = torch.zeros_like(x_init_MC)
    
    Yl,Yh = wutils.wave_mat2list(wavelet_mat,level)
    
    alpha = [None] * (3*level + 1)
    wavelet_mat_jittered = wavelet_mat.clone()
    eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances))
    
    eta = torch.max(eta1,eta2)
    eps = torch.tensor(2.22e-16, device = device)
    eta = eta + eps 
    
    noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),0)

    
    wavelet_mat_jittered += eta * noise_vec
    
    denoised_jittered = denoiser(wavelet_mat_jittered, variances, x_init_MC[0,:,:,:].unsqueeze(0))
    
    next_x_warm_MC[0,:,:,:] = (denoised_jittered.squeeze(0)).clone()
    
    alpha[0] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(((denoised_jittered - denoised)/eta).permute(0,2,3,1).contiguous()).reshape(-1))))

    

    count = 1
    for s in range(level):
        index = level - 1 - s
        for b in range(3):
            
            wavelet_mat_jittered = wavelet_mat.clone()
            eta1 = torch.max(transforms_new.complex_abs(Yh[index][:,:,b,:,:].permute(0,2,3,1)))/1000.
            eta = torch.max(eta1,eta2)
            eps = torch.tensor(2.22e-16, device = device)
            eta = eta + eps
            
            noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),count)

            wavelet_mat_jittered += eta * noise_vec

            denoised_jittered = denoiser(wavelet_mat_jittered, variances, x_init_MC[count,:,:,:].unsqueeze(0))
            next_x_warm_MC[count,:,:,:] = (denoised_jittered.squeeze(0)).clone()
            
            
            alpha[count] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(((denoised_jittered - denoised)/eta).permute(0,2,3,1).contiguous()).reshape(-1))))

            count = count + 1 
                    
    return torch.tensor(alpha, device = device), next_x_warm_MC



def calc_batch_MC_divergence_complex_with_warm_strt(denoiser, wavelet_mat, variances, x_init_with_MC, level,subband_sizes,p1m1_mask):

    """This is for processing multiple subband batches only
    """
    device = wavelet_mat.device
    
    next_x_warm_MC = torch.zeros_like(x_init_with_MC)
    
#     Yl,Yh = wutils.wave_mat2list(wavelet_mat,level)
    
    alpha = torch.zeros(3*level + 1, device = device)

#     wavelet_mat_jittered = wavelet_mat.clone()
    
#     eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances))
#     eta2 = torch.max(torch.sqrt(variances))
    
#     eta = torch.max(eta1,eta2)
    eta = eta2
    eps = torch.tensor(2.22e-16, device = device)
    eta = eta + eps 
    
    batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch(wavelet_mat,level,eta,p1m1_mask)
    
#     print("variances ")
#     print(variances)
    
    
    denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances, x_init_with_MC)

    next_x_warm_with_MC = (denoised_jittered_with_denoised).clone()
    denoised = denoised_jittered_with_denoised[0,:,:,:].unsqueeze(0)
    denoised_jittered = denoised_jittered_with_denoised[1:,:,:,:]
#     print("denoised ")
#     print(torch.sum(torch.isnan(denoised)))
#     print("denoised jittered")
#     print(torch.sum(torch.isnan(denoised_jittered)))
    
    alpha = (1. / subband_sizes)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),((denoised_jittered - denoised)/eta).permute(0,2,3,1))
    
    return denoised, alpha, next_x_warm_with_MC



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






def calc_batch_MC_divergence_complex(denoiser, wavelet_mat, variances, level,subband_sizes,p1m1_mask):

    """This is for processing multiple subband batches only
    """
    device = wavelet_mat.device
           
    alpha = torch.zeros(3*level + 1, device = device)

    
#     eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances))
    
#     eta = torch.max(eta1,eta2)
    eta = eta2
    eps = torch.tensor(2.22e-16, device = device)
    eta = eta + eps 
    
    batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch(wavelet_mat,level,eta,p1m1_mask)
    
    denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances)

    denoised = denoised_jittered_with_denoised[0,:,:,:].unsqueeze(0)
    denoised_jittered = denoised_jittered_with_denoised[1:,:,:,:]
    
    alpha = (1. / subband_sizes)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),((denoised_jittered - denoised)/eta).permute(0,2,3,1))
    
    return denoised, alpha


def calc_batch_MC_divergence_true_complex(denoiser, wavelet_mat, variances, level,subband_sizes,p1m1_mask, ifm, scale_percentile):

    """This is for processing multiple subband batches only
    """
    # Find the scaling factor
    
    noisy_image = wutils.wave_inverse_mat(wavelet_mat, ifm, level)
    sorted_image_vec = transforms_new.complex_abs(noisy_image.squeeze(0).permute(1,2,0)).reshape((-1,)).sort()
    scale = sorted_image_vec.values[int(len(sorted_image_vec.values) * scale_percentile/100)].item()
    
    device = wavelet_mat.device
           
    alpha = torch.zeros(3*level + 1, device = device)

    
#     eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances))
    
#     eta = torch.max(eta1,eta2)
    eta = eta2
    eps = torch.tensor(2.22e-16, device = device)
    eta = eta + eps 
    
    batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch(wavelet_mat,level,eta,p1m1_mask)
    
    denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances, scale)

    denoised = denoised_jittered_with_denoised[0,:,:,:].unsqueeze(0)
    denoised_jittered = denoised_jittered_with_denoised[1:,:,:,:]
    
    alpha = (1. / subband_sizes)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),((denoised_jittered - denoised)/eta).permute(0,2,3,1))
    
    return denoised, alpha



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
    
#     print('LMSE Stage delta selected pos --> eta2 (var), neg --> eta 1 (abs (r))')
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




# Incomplete- probably not necessary 
# def calc_batch_MC_divergence_true_complex_with_sens_map_mask(denoiser, wavelet_mat, variances, level,masked_subband_sizes,p1m1_mask, ifm, scale_percentile):

#     """This is for processing multiple subband batches only
#     """
#     # Find the scaling factor
    
#     noisy_image = wutils.wave_inverse_mat(wavelet_mat, ifm, level)
#     sorted_image_vec = transforms_new.complex_abs(noisy_image.squeeze(0).permute(1,2,0)).reshape((-1,)).sort()
#     scale = sorted_image_vec.values[int(len(sorted_image_vec.values) * scale_percentile/100)].item()
    
#     device = wavelet_mat.device
           
#     alpha = torch.zeros(3*level + 1, device = device)

    
# #     eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
#     eta2 = torch.mean(torch.sqrt(variances))
    
# #     eta = torch.max(eta1,eta2)
#     eta = eta2
#     eps = torch.tensor(2.22e-16, device = device)
#     eta = eta + eps 
    
#     batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch_with_sens_map_mask(wavelet_mat,level,eta,p1m1_mask,sens_map_mask)
    
#     denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances, scale)

#     denoised = denoised_jittered_with_denoised[0,:,:,:].unsqueeze(0)
#     denoised_jittered = denoised_jittered_with_denoised[1:,:,:,:]
    
#     alpha = (1. / subband_sizes)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),((denoised_jittered - denoised)/eta).permute(0,2,3,1))
    
#     return denoised, alpha






def calc_batch_2_MC_divergence_complex_with_warm_strt(denoiser, wavelet_mat, variances, x_init_with_MC, level,subband_sizes_batch,p1m1_mask_batch):

    """This is for processing multiple image batches and also subband batches
    """
    device = wavelet_mat.device
    batch_len = wavelet_mat.shape[0]
    
    next_x_warm_MC = torch.zeros_like(x_init_with_MC)
    
#     Yl,Yh = wutils.wave_mat2list(wavelet_mat,level)
    
    alpha = torch.zeros(batch_len*(3*level + 1), device = device)

#     wavelet_mat_jittered = wavelet_mat.clone()
    
#     eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances),1)
    eta2_batch = eta2.repeat_interleave(3*level + 1)
    
#     eta = torch.max(eta1,eta2)
    eta_batch = eta2_batch
    eps = torch.tensor(2.22e-16, device = device)
    eta_batch = eta_batch + eps 
    
    batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch_2(wavelet_mat,level,eta_batch,p1m1_mask_batch)
    
    variances_new = torch.zeros(batch_len*(3*level + 2),3*level + 1,device = device)
    variances_new[0:batch_len,:] = variances.clone()
    variances_new[batch_len:,:] = variances.clone().repeat_interleave(3*level + 1,dim = 0)
    
    
    denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances_new, x_init_with_MC)

    next_x_warm_with_MC = (denoised_jittered_with_denoised).clone()
    
    denoised = denoised_jittered_with_denoised[0:batch_len,:,:,:]
    denoised_jittered = denoised_jittered_with_denoised[batch_len:,:,:,:]
    denoised_interleaved = denoised.clone().repeat_interleave(3*level + 1,dim = 0)
    
    alpha = (1. / subband_sizes_batch)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),(((denoised_jittered - denoised_interleaved).permute(1,2,3,0)/eta_batch).permute(3,0,1,2)).permute(0,2,3,1))
    
    return denoised, alpha.reshape(batch_len,3*level + 1), next_x_warm_with_MC




def calc_batch_2_MC_divergence_complex(denoiser, wavelet_mat, variances, level,subband_sizes_batch,p1m1_mask_batch):
    
    """This is for processing multiple image batches and also subband batches
    """
    device = wavelet_mat.device
    batch_len = wavelet_mat.shape[0]
    
    alpha = torch.zeros(batch_len*(3*level + 1), device = device)

    
#     eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances),1)
    eta2_batch = eta2.repeat_interleave(3*level + 1)
    
#     eta = torch.max(eta1,eta2)
    eta_batch = eta2_batch
    eps = torch.tensor(2.22e-16, device = device)
    eta_batch = eta_batch + eps 
    
    batch_wave_mat_jittered, noise_mat = wutils.get_my_jittred_batch_2(wavelet_mat,level,eta_batch,p1m1_mask_batch)
    
    variances_new = torch.zeros(batch_len*(3*level + 2),3*level + 1,device = device)
    variances_new[0:batch_len,:] = variances.clone()
    variances_new[batch_len:,:] = variances.clone().repeat_interleave(3*level + 1,dim = 0)
    
    denoised_jittered_with_denoised = denoiser(batch_wave_mat_jittered, variances_new)

    denoised = denoised_jittered_with_denoised[0:batch_len,:,:,:]
    denoised_jittered = denoised_jittered_with_denoised[batch_len:,:,:,:]
    denoised_interleaved = denoised.clone().repeat_interleave(3*level + 1,dim = 0)
    
#     alpha = (1. / subband_sizes_batch)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),((denoised_jittered - denoised_interleaved)/eta_batch).permute(0,2,3,1))

    alpha = (1. / subband_sizes_batch)*transforms_new.real_part_of_aHb(noise_mat.permute(0,2,3,1),(((denoised_jittered - denoised_interleaved).permute(1,2,3,0)/eta_batch).permute(3,0,1,2)).permute(0,2,3,1))
    
    return denoised, alpha.reshape(batch_len,3*level + 1)






class LMSE_CG_GEC_with_div_and_warm_strt:
    """Wrapper of LMSE for using with GEC which used CG. This code uses warm start"""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w, xfm, ifm,level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4)):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sigma_w = sigma_w
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        self.xfm = xfm
        self.ifm = ifm
        
        
    def __call__(self, wavelet_mat, variances, x_init, x_init_MC, calc_divergence=True):

        variances *= self.beta_tune_LMSE
       
        denoised = self._denoise(wavelet_mat, variances, x_init)
        if calc_divergence:
            alpha, next_x_warm_MC = calc_MC_divergence_complex_with_warm_strt(self._denoise, denoised, wavelet_mat, variances, x_init_MC, self.level)
            return denoised, alpha , next_x_warm_MC
        else:
            return denoised

    def _denoise(self, wavelet_mat, variances, x_init):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = (self.sigma_w**2)*(GAMMA_1_full)
        
        B_op_foo = B_op(self.idx1_complement,self.idx2_complement, self.xfm, self.ifm, self.level)       
        A_bar = CG_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_op_foo = CG_INP_op(B_op_foo,self.level,my_scalar_vec) 
        b_bar = CG_INP_op_foo.forward(self.y,wavelet_mat)
        
#         x_init = torch.zeros_like(wavelet_mat)
#         x_init = torch.randn_like(wavelet_mat)
#         x_init = B_op_foo.H(self.y)
        
        denoised_wavelet,stop_iter,rtr_end = CG_method(b_bar,A_bar,x_init,self.LMSE_inner_iter_lim,self.eps_lim) 

        return denoised_wavelet

    
    
    

class LMSE_batch_CG_GEC_with_div_and_warm_strt:
    """Wrapper of LMSE for using with GEC which used CG. This code uses warm start"""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w, xfm,ifm, p1m1_mask, subband_sizes, level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4)):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sigma_w = sigma_w
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        self.subband_sizes = subband_sizes
        self.p1m1_mask = p1m1_mask
        self.xfm = xfm
        self.ifm = ifm
        
    def __call__(self, wavelet_mat, variances, x_init_with_MC, calc_divergence=True):

        variances *= self.beta_tune_LMSE
        
        denoised, alpha, next_x_warm_with_MC = calc_batch_MC_divergence_complex_with_warm_strt(self._denoise, wavelet_mat, variances, x_init_with_MC, self.level,self.subband_sizes,self.p1m1_mask)
        
        return denoised, alpha, next_x_warm_with_MC


    def _denoise(self, wavelet_mat, variances, x_init):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = (self.sigma_w**2)*(GAMMA_1_full)
        
        B_op_foo = B_op(self.idx1_complement,self.idx2_complement, self.xfm, self.ifm, self.level)       
        A_bar = CG_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_op_foo = CG_INP_op(B_op_foo,self.level,my_scalar_vec) 
        b_bar = CG_INP_op_foo.forward(self.y,wavelet_mat)

        denoised_wavelet,stop_iter,rtr_end = CG_method_subbands(b_bar,A_bar,x_init,self.LMSE_inner_iter_lim) 

        return denoised_wavelet
    
    
class LMSE_batch_CG_GEC_with_div_and_warm_strt_DC:
    """Wrapper of LMSE for using with GEC which used CG. This code uses warm start"""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w, xfm,ifm, p1m1_mask, subband_sizes, Dmh, level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4)):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sigma_w = sigma_w
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        self.subband_sizes = subband_sizes
        self.p1m1_mask = p1m1_mask
        self.xfm = xfm
        self.ifm = ifm
        self.Dmh = Dmh
        
    def __call__(self, wavelet_mat, variances, x_init_with_MC, calc_divergence=True):

        variances *= self.beta_tune_LMSE
        
        denoised, alpha, next_x_warm_with_MC = calc_batch_MC_divergence_complex_with_warm_strt(self._denoise, wavelet_mat, variances, x_init_with_MC, self.level,self.subband_sizes,self.p1m1_mask)
        
        return denoised, alpha, next_x_warm_with_MC


    def _denoise(self, wavelet_mat, variances, x_init):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = (self.sigma_w**2)*(GAMMA_1_full)
        
        B_op_foo = B_op_DC(self.idx1_complement,self.idx2_complement, self.xfm, self.ifm, self.level, self.Dmh)       
        A_bar = CG_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_op_foo = CG_INP_op(B_op_foo,self.level,my_scalar_vec) 
        b_bar = CG_INP_op_foo.forward(self.y,wavelet_mat)

        denoised_wavelet,stop_iter,rtr_end = CG_method_subbands(b_bar,A_bar,x_init,self.LMSE_inner_iter_lim) 

        return denoised_wavelet
    
    
    
    
    

class LMSE_batch_2_CG_GEC_with_div_and_warm_strt:
    """Wrapper of LMSE for using with GEC which used CG. This code uses warm start. This works for image batches and subband batches"""
    def __init__(self, y_batch, idx1_complement, idx2_complement,sigma_w_batch, xfm,ifm, p1m1_mask_batch, subband_sizes_batch, level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4)):

        self.y_batch = y_batch
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sigma_w_batch = sigma_w_batch
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        self.subband_sizes_batch = subband_sizes_batch
        self.p1m1_mask_batch = p1m1_mask_batch
        self.xfm = xfm
        self.ifm = ifm
        
    def __call__(self, wavelet_mat, variances, x_init_with_MC, calc_divergence=True):

        variances *= self.beta_tune_LMSE
        
        denoised, alpha, next_x_warm_with_MC = calc_batch_2_MC_divergence_complex_with_warm_strt(self._denoise, wavelet_mat, variances, x_init_with_MC, self.level,self.subband_sizes_batch,self.p1m1_mask_batch)
        
        return denoised, alpha, next_x_warm_with_MC


    def _denoise(self, wavelet_mat, variances, x_init):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = ((self.sigma_w_batch**2)*(GAMMA_1_full.permute(1,0))).permute(1,0)
        
        B_op_foo = B_op(self.idx1_complement,self.idx2_complement, self.xfm, self.ifm, self.level)       
        A_bar = CG_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_op_foo = CG_INP_op(B_op_foo,self.level,my_scalar_vec) 
        b_bar = CG_INP_op_foo.forward(self.y_batch,wavelet_mat)

        denoised_wavelet,stop_iter,rtr_end = CG_method_subbands(b_bar,A_bar,x_init,self.LMSE_inner_iter_lim) 

        return denoised_wavelet

    
    

class LMSE_batch_CG_GEC_with_div_and_warm_strt_multi_coil:
    """Wrapper of LMSE for using with GEC which used CG. This code uses warm start. This is for multicoil"""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w, xfm,ifm, p1m1_mask, subband_sizes, sense_map, level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4), changeFactor = 0.1):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        self.subband_sizes = subband_sizes
        self.p1m1_mask = p1m1_mask
        self.xfm = xfm
        self.ifm = ifm
        self.sense_map = sense_map
        self.changeFactor = changeFactor
        self.sigma_w = sigma_w/torch.sqrt(self.beta_tune_LMSE)
        
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

        denoised_wavelet,stop_iter,rtr_end = CG_method_subbands(b_bar,A_bar,x_init,self.LMSE_inner_iter_lim) 

        return denoised_wavelet
    
    
    
    
    
    
    
class LMSE_CG_GEC_with_div:
    """Wrapper of LMSE for using with GEC which used CG."""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w, xfm, ifm, level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4)):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sigma_w = sigma_w
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        self.xfm = xfm
        self.ifm = ifm
        
    def __call__(self, wavelet_mat, variances, calc_divergence=True):

        variances *= self.beta_tune_LMSE
       
        denoised = self._denoise(wavelet_mat, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_complex(self._denoise, denoised, wavelet_mat, variances, self.level)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet_mat, variances):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = (self.sigma_w**2)*(GAMMA_1_full)
        
        B_op_foo = B_op(self.idx1_complement,self.idx2_complement, self.xfm, self.ifm, self.level)       
        A_bar = CG_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_op_foo = CG_INP_op(B_op_foo,self.level,my_scalar_vec) 
        b_bar = CG_INP_op_foo.forward(self.y,wavelet_mat)
        
        x_init = torch.zeros_like(wavelet_mat)
#         x_init = torch.randn_like(wavelet_mat)
#         x_init = B_op_foo.H(self.y)
        
        denoised_wavelet,stop_iter,rtr_end = CG_method(b_bar,A_bar,x_init,self.LMSE_inner_iter_lim,self.eps_lim) 

        return denoised_wavelet
    
    
    
    
class LMSE_CG_GEC_precond_with_div:
    """Wrapper of preconditioned LMSE for using with GEC which used CG."""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w, xfm, ifm, level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4)):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sigma_w = sigma_w
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        self.xfm = xfm
        self.ifm = ifm
        
    def __call__(self, wavelet_mat, variances, calc_divergence=True):

        variances *= self.beta_tune_LMSE
       
        denoised = self._denoise(wavelet_mat, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_complex(self._denoise, denoised, wavelet_mat, variances, self.level)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet_mat, variances):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = torch.pow((self.sigma_w**2)*(GAMMA_1_full),-0.5)
        
        B_op_foo = B_op(self.idx1_complement,self.idx2_complement, self.xfm, self.ifm, self.level)       
        A_precond = CG_precond_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_precond_op_foo = CG_INP_precond_op(B_op_foo,self.level,my_scalar_vec)
        b_precond = CG_INP_precond_op_foo.forward(self.y,wavelet_mat)
        
        x_init = torch.zeros_like(wavelet_mat)
#         x_init = torch.randn_like(wavelet_mat)
#         x_init = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(B_op_foo.H(self.y),level = self.level),my_scalar_vec)) 
    
        denoised_wavelet_pre,stop_iter,rtr_end = CG_method(b_precond,A_precond,x_init,self.LMSE_inner_iter_lim,self.eps_lim) 
        
        denoised_wavelet = wutils.get_wave_mat(wutils.wave_scalar_mul_subbandwise_list(wutils.wave_mat2list(denoised_wavelet_pre,level = self.level),my_scalar_vec))
        
        return denoised_wavelet
    

def correct_GAMMA_using_EM_with_mask_LMSE(denoiser,noisy_wavelet_mat, variances, warm_start_mat, num_em_iter, level, wave_mask_list):

    """Correct the variances entering the denoiser using EM iterations

    """
    device = noisy_wavelet_mat.device
#     print('running EM correction for LMSE')
    
    for i in range(num_em_iter):
                
        denoised, alpha, warm_start_mat = denoiser(noisy_wavelet_mat, variances,warm_start_mat,True)

        variances = torch.as_tensor(wutils.find_subband_wise_MSE_list_with_wave_mask(wutils.wave_mat2list(denoised,level),wutils.wave_mat2list(noisy_wavelet_mat,level), wave_mask_list), device = device)+ variances*torch.as_tensor(alpha, device = device)
        
    return variances



def correct_GAMMA_using_EM_LMSE(denoiser,noisy_wavelet_mat, variances, warm_start_mat, num_em_iter, level):

    """Correct the variances entering the denoiser using EM iterations

    """
    device = noisy_wavelet_mat.device
#     print('running EM correction for LMSE')
    
    for i in range(num_em_iter):
                
        denoised, alpha, warm_start_mat = denoiser(noisy_wavelet_mat, variances,warm_start_mat,True)

        variances = torch.as_tensor(wutils.find_subband_wise_MSE_list(wutils.wave_mat2list(denoised,level),wutils.wave_mat2list(noisy_wavelet_mat,level)), device = device)+ variances*torch.as_tensor(alpha, device = device)
        
    return variances





def correct_GAMMA_using_EM(denoiser,noisy_wavelet_mat, variances, num_em_iter, level):

    """Correct the variances entering the denoiser using EM iterations

    """
    device = noisy_wavelet_mat.device
#     print('running EM correction')
    
    for i in range(num_em_iter):
                
        denoised, alpha = denoiser(noisy_wavelet_mat, variances)
        
        variances = torch.as_tensor(wutils.find_subband_wise_MSE_list(wutils.wave_mat2list(denoised,level),wutils.wave_mat2list(noisy_wavelet_mat,level)), device = device)+ variances*torch.as_tensor(alpha, device = device)
        
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





def correct_GAMMA_using_EM_batch(denoiser,noisy_wavelet_mat, variances, num_em_iter, level):

    """Correct the variances entering the denoiser using EM iterations

    """
    device = noisy_wavelet_mat.device
#     print('running EM correction')
    
    for i in range(num_em_iter):
                
        denoised, alpha = denoiser(noisy_wavelet_mat, variances)
        
        variances = torch.as_tensor(wutils.find_subband_wise_MSE_list_batch(wutils.wave_mat2list(denoised,level),wutils.wave_mat2list(noisy_wavelet_mat,level)), device = device)+ variances*torch.as_tensor(alpha, device = device)
        
    return variances

class DnCNN_cpc_VDAMP_complex:
    """
    """
    def __init__(self, modeldir, std_ranges,xfm,ifm, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = torch.tensor(1), complex_weight = torch.tensor(0.1), device=torch.device('cpu'),
                std_pool_func=torch.mean, verbose=False, level = 4):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 10 / 255.
                denoiser 2 is for noise with std 10 to 20 / 255.
                denoiser 3 is for noise with std 20 / 255 to 50 / 255.
                denoiser 4 is for noise with std 50 / 255 to 120 / 255.
                denoiser 5 is for noise with std 120 / 255 to 500 / 255.
            channels (int): number of channels in the model.
            wavetype (str): type of wavelet transform.
            num_layers (int): number of layers in the model.
            std_channels (int): number of std channels for the model i.e.
                number of wavelet subbands.
            device: the device to run the model on.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.channels = channels
        self.std_ranges = std_ranges
        self.wavetype = wavetype
        self.device = device
        self.models = self._load_models(modeldir)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.beta_tune = beta_tune
        self.complex_weight = complex_weight
        self.level = level
        self.xfm = xfm
        self.ifm = ifm

    def __call__(self, wavelet_mat, variances, gamma=torch.tensor(1.0), calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.
            gamma (float): scaling on the variances.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances) * gamma
        variances *= self.beta_tune 
        denoised = self._denoise(wavelet_mat, variances)
        if calc_divergence:
            
            alpha = calc_MC_divergence_complex(self._denoise, denoised, wavelet_mat, variances, self.level)
            
            return denoised, alpha
        else:
            return denoised
    
    
    @torch.no_grad()
    def _denoise(self, wavelet_mat, variances):

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
        
        if self.channels == 1:
            
            
            
            noisy_real = noisy_image[:,0,:,:].unsqueeze(1)
            noisy_imag = noisy_image[:,1,:,:].unsqueeze(1)
            
            model_inp = (noisy_image).clone()

            
            model_inp[:,1,:,:] = wutils.wave_inverse_list(wutils.add_noise_subbandwise_list(wutils.wave_mat2list(torch.zeros_like(noisy_real), self.level),stds),self.ifm)
            
            

            denoised_real = self.models[select](model_inp)
            
            print("hello")
            dummy = wutils.wave_forward_mat(noisy_image,self.xfm)
            dummy = wutils.wave_forward_mat(denoised_real.detach(),self.xfm)
            print("hello")
            
            
            denoised_imag = noisy_imag * self.complex_weight
            
            denoised_image = torch.zeros_like(noisy_image)
            denoised_image[:,0,:,:] = denoised_real[:,0,:,:]
            denoised_image[:,1,:,:] = denoised_imag[:,0,:,:]
            
#             print(denoised_image.shape)
#             print('normalizing')
#             denoised_image[0] = denoised_image[0]/torch.max(torch.abs(denoised_image[0]))

            denoised_wavelet_mat = wutils.wave_forward_mat(denoised_image,self.xfm)
        else:
            raise ValueError('Only support channel == 1')
        return denoised_wavelet_mat

    def _load_models(self, modeldirs):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = load_model(modeldir)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    
    
    
class DnCNN_cpc_VDAMP_complex_batch:
    """
    """
    def __init__(self, modeldir, std_ranges, xfm,ifm, p1m1_mask, subband_sizes,  channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = torch.tensor(1), complex_weight = torch.tensor(0.1), device=torch.device('cpu'),
                std_pool_func=torch.mean, verbose=False, level = 4):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 10 / 255.
                denoiser 2 is for noise with std 10 to 20 / 255.
                denoiser 3 is for noise with std 20 / 255 to 50 / 255.
                denoiser 4 is for noise with std 50 / 255 to 120 / 255.
                denoiser 5 is for noise with std 120 / 255 to 500 / 255.
            channels (int): number of channels in the model.
            wavetype (str): type of wavelet transform.
            num_layers (int): number of layers in the model.
            std_channels (int): number of std channels for the model i.e.
                number of wavelet subbands.
            device: the device to run the model on.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.channels = channels
        self.std_ranges = std_ranges
        self.wavetype = wavetype
        self.device = device
        self.models = self._load_models(modeldir)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.beta_tune = beta_tune
        self.complex_weight = complex_weight
        self.level = level
        self.subband_sizes = subband_sizes
        self.p1m1_mask = p1m1_mask
        self.xfm = xfm
        self.ifm = ifm
        
    def __call__(self, wavelet_mat, variances, calc_divergence=True):

        variances *= self.beta_tune
        variances *= 0.5 # We do this since the denoiser is real image denoiser and hence we need to give the denoiser the variance in the real channel (full image variance = real channel variance + imaginary channel variance, we assume (experimentally verified) the real and imaginary channel vairance is approximately the same.)
        
        denoised, alpha = calc_batch_MC_divergence_complex(self._denoise, wavelet_mat, variances, self.level,self.subband_sizes,self.p1m1_mask)
        
        return denoised, alpha
    
    
    @torch.no_grad()
    def _denoise(self, wavelet_mat, variances):

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
        
        if self.channels == 1:
            
            
            
            noisy_real = noisy_image[:,0,:,:].unsqueeze(1)
            noisy_imag = noisy_image[:,1,:,:].unsqueeze(1)
            
            model_inp = (noisy_image).clone()

            
#             model_inp[:,1:,:,:] = wutils.wave_inverse_mat(wutils.get_wave_mat(wutils.add_noise_subbandwise_list(wutils.wave_mat2list(torch.zeros_like(noisy_real), self.level),stds)),self.ifm,self.level)

            model_inp[:,1:,:,:] = wutils.wave_inverse_list(wutils.add_noise_subbandwise_list(wutils.wave_mat2list(torch.zeros_like(noisy_real), self.level),stds),self.ifm)
            
#             ttt = time.time()
            denoised_real = self.models[select](model_inp)
#             print("elapsed: ", time.time() - ttt)

            dummy = wutils.wave_forward_mat(noisy_image[0:1,:,:,:],self.xfm)
#             dummy = wutils.wave_forward_mat(denoised_real.detach(),self.xfm)


                
#             denoised_imag = noisy_imag * self.complex_weight
#             denoised_image = torch.zeros_like(noisy_image)
#             denoised_image[:,0,:,:] = denoised_real[:,0,:,:]
#             denoised_image[:,1,:,:] = denoised_imag[:,0,:,:]
            
            
            denoised_image = self.complex_weight*noisy_image.clone()
            denoised_image[:,0,:,:] = denoised_real[:,0,:,:]
#             denoised_image[:,1,:,:] = denoised_imag[:,0,:,:]            

            
#             print(denoised_image.shape)
#             print('normalizing')
#             denoised_image[0] = denoised_image[0]/torch.max(torch.abs(denoised_image[0]))
    
            denoised_wavelet_mat = wutils.wave_forward_mat(denoised_image,self.xfm)

            
        else:
            raise ValueError('Only support channel == 1')
        return denoised_wavelet_mat

    def _load_models(self, modeldirs):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = load_model(modeldir)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    

    
class DnCNN_cpc_VDAMP_complex_batch_2:
    """ Works for image and subband batches
    """
    def __init__(self, modeldir, std_ranges, xfm,ifm, p1m1_mask_batch, subband_sizes_batch,  channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = torch.tensor(1), complex_weight = torch.tensor(0.1), device=torch.device('cpu'),
                std_pool_func=torch.mean, verbose=False, level = 4):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 10 / 255.
                denoiser 2 is for noise with std 10 to 20 / 255.
                denoiser 3 is for noise with std 20 / 255 to 50 / 255.
                denoiser 4 is for noise with std 50 / 255 to 120 / 255.
                denoiser 5 is for noise with std 120 / 255 to 500 / 255.
            channels (int): number of channels in the model.
            wavetype (str): type of wavelet transform.
            num_layers (int): number of layers in the model.
            std_channels (int): number of std channels for the model i.e.
                number of wavelet subbands.
            device: the device to run the model on.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.channels = channels
        self.std_ranges = std_ranges
        self.wavetype = wavetype
        self.device = device
        self.models = self._load_models(modeldir)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.beta_tune = beta_tune
        self.complex_weight = complex_weight
        self.level = level
        self.subband_sizes_batch = subband_sizes_batch
        self.p1m1_mask_batch = p1m1_mask_batch
        self.xfm = xfm
        self.ifm = ifm
        
    def __call__(self, wavelet_mat, variances, calc_divergence=True):

        variances *= self.beta_tune
        
        denoised, alpha = calc_batch_2_MC_divergence_complex(self._denoise, wavelet_mat, variances, self.level,self.subband_sizes_batch,self.p1m1_mask_batch)
        
        return denoised, alpha
    
    
    @torch.no_grad()
    def _denoise(self, wavelet_mat, variances):

        # Select the model to use
        stds = torch.sqrt(variances)
        std_pooled = self.std_pool_func(torch.sqrt(variances),1)
        
        select_batch = (torch.sum(std_pooled.reshape(len(std_pooled),1).repeat(1,len(self.std_ranges)) > self.std_ranges.reshape(1,len(self.std_ranges)).repeat(len(std_pooled),1),1)) - 1

        select_batch = torch.clip(select_batch,0, len(self.models) - 1)
        
        indices = []
        for i in range(len(self.models)):
            indices.append(torch.where(select_batch==i))

        # Denoise
        noisy_image = wutils.wave_inverse_mat(wavelet_mat, self.ifm, self.level)
        
        if self.channels == 1:
            
            model_inp = (noisy_image).clone()
            denoised_real = torch.zeros_like(noisy_image)
            
            model_inp[:,1:,:,:] = wutils.wave_inverse_list(wutils.add_noise_subbandwise_list_batch(wutils.wave_mat2list(torch.zeros_like(noisy_image[:,0:1,:,:]), self.level),stds),self.ifm)
            
            for i in range(len(self.models)):
                if len(indices[i][0]) > 0:
                    denoised_real[indices[i][0],:,:,:] = self.models[i](model_inp[indices[i][0],:,:,:])

            dummy = wutils.wave_forward_mat(noisy_image[0:1,:,:,:],self.xfm)
            
            denoised_image = self.complex_weight*noisy_image.clone()
            denoised_image[:,0,:,:] = denoised_real[:,0,:,:]

            denoised_wavelet_mat = wutils.wave_forward_mat(denoised_image,self.xfm)
            
        else:
            raise ValueError('Only support channel == 1')
        return denoised_wavelet_mat

    def _load_models(self, modeldirs):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = load_model(modeldir)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models


    
    
class DnCNN_cpc_VDAMP_true_complex_batch:
    """
    handles true complex denoising; used mainly for fastMRI data; 
    """
    def __init__(self, modeldir, std_ranges, xfm,ifm, p1m1_mask, subband_sizes,  channels=2, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = torch.tensor(1), device=torch.device('cpu'),
                std_pool_func=torch.mean, verbose=False, level = 4, scale_percentile = 98, changeFactor = 0.1):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 10 / 255.
                denoiser 2 is for noise with std 10 to 20 / 255.
                denoiser 3 is for noise with std 20 / 255 to 50 / 255.
                denoiser 4 is for noise with std 50 / 255 to 120 / 255.
                denoiser 5 is for noise with std 120 / 255 to 500 / 255.
            channels (int): number of channels in the model.
            wavetype (str): type of wavelet transform.
            num_layers (int): number of layers in the model.
            std_channels (int): number of std channels for the model i.e.
                number of wavelet subbands.
            device: the device to run the model on.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
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
        
#         denoised, alpha = calc_batch_MC_divergence_true_complex(self._denoise, wavelet_mat, variances, self.level,self.subband_sizes,self.p1m1_mask, self.ifm, self.scale_percentile)
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

    
    
    
    
    

    
    
# def calc_MC_LMSE_divergence_complex(denoiser, denoised, wavelet_mat, variances, level):
# #     print('complex divergence computation +-1 for LMSE!')
#     # modified by Saurav

#     alpha = [None] * (3*level + 1)
    
#     wavelet_mat_jittered = torch.zeros_like(wavelet_mat)

#     noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),0)
        
#     wavelet_mat_jittered += noise_vec

#     denoised_jittered = denoiser(wavelet_mat_jittered, variances, torch.zeros_like(wavelet_mat))
    
#     alpha[0] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(denoised_jittered.permute(0,2,3,1).contiguous()).reshape(-1))))

#     count = 1

#     for s in range(level):
        
#         for b in range(3):

#             wavelet_mat_jittered = torch.zeros_like(wavelet_mat)

#             noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),count)
            
#             wavelet_mat_jittered += noise_vec

#             denoised_jittered = denoiser(wavelet_mat_jittered, variances, torch.zeros_like(wavelet_mat))
            
#             alpha[count] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(denoised_jittered.permute(0,2,3,1).contiguous()).reshape(-1))))

#             count = count + 1
                    
#     return alpha


# class LMSE_GEC_pre_conditioned_div:
#     """Wrapper of LMSE for using with GEC."""
#     def __init__(self, y, idx1_complement, idx2_complement,sigma_w,level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-10)):

#         self.y = y
#         self.idx1_complement = idx1_complement
#         self.idx2_complement = idx2_complement
#         self.sigma_w = sigma_w
#         self.level = level
#         self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
#         self.beta_tune_LMSE = beta_tune_LMSE
#         self.eps_lim = eps_lim
        
#     def __call__(self, wavelet_mat, variances, calc_divergence=True):

#         variances *= self.beta_tune_LMSE
       
#         denoised = self._denoise(wavelet_mat, variances,self.y)
#         if calc_divergence:
#             alpha = calc_MC_LMSE_divergence_complex(self._denoise, denoised, wavelet_mat, variances, self.level)
#             return denoised, alpha
#         else:
#             return denoised

#     def _denoise(self, wavelet_mat, variances, y_denoiser):
        
#         GAMMA_1_full = 1/variances
#         my_scalar_vec = self.sigma_w*torch.sqrt(GAMMA_1_full)
        
#         B_op_foo = B_op(self.idx1_complement,self.idx2_complement)       
#         A_precon = CG_op(B_op_foo,self.level,my_scalar_vec)
        
#         CG_INP_op_foo = CG_INP_op(self.level,my_scalar_vec)
#         b_precon = CG_INP_op_foo.forward(y_denoiser,wavelet_mat)
        
#         x_init = torch.zeros_like(wavelet_mat)
        
#         denoised_wavelet,stop_iter,rtr_end = CG_method(b_precon,A_precon,x_init,self.LMSE_inner_iter_lim,self.eps_lim) # Cannot use CG method here since the matrix is not square and hermetian symmetric

#         return denoised_wavelet




