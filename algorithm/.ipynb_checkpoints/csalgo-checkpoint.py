"""Compressive sensing reconstruction algorithms.

    * DIT
    * DAMP

Notes:
    It is the client's resposibility to call these functions under
    torch.no_grad() environment where appropriate.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
from utils.general import calc_psnr, generate_noise

class IterativeDenoisingCSRecon:
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image,
                 verbose):
        """Initialize an iterative denoising CS solver.

        Args:
            image_shape (list/array): shape of the ground truth image in the format (C, H, W).
            denoiser: the denoiser for thresholding.
            iters: number of iterations.
            image: the ground truth image. If image is given, calculate the PSNR of the 
                thresholding result at every iteration.
            verbose (bool): whether to print standard deviation of the effective noise 
                and/or PSNR at every iteration.
        """
        self.H = image_shape[1]
        self.W = image_shape[2]
        self.denoiser = denoiser
        self.iters = iters
        self.image = image
        self.verbose = verbose

    def __call__(self, y, Afun, Atfun, m, n):
        """Solve the CS reconstruction problem.

        Args:
            y (tensor): the measurement with dimension m.
            Afun: the forward CS measurement operator.
            Atfun: the transpose of Afun.
            m (int): dimension of the measurement.
            n (int): dimension of the ground truth.

        Returns:
            output: CS reconstruction result.
            psnr: PSNR of the thresholding results at every iteration.
            r_t: The noisy image before thresholding at the last iteration.
            sigma_hat_t: the estimated standard deviation of the effective noise at the last iteration.
        """
        pass


class DIT(IterativeDenoisingCSRecon):
    """CS solver with Denoising-based Iterative Thresholding.

    Note:
        DIT is similar to DAMP, but without the Onsager correction.
    """
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image=None,
                 verbose=False):
        super().__init__(image_shape, denoiser, iters,
                         image, verbose)

    def __call__(self, y, Afun, Atfun, m, n):
        psnr = _setup(self)
        x_t = torch.zeros(n, 1)
        z_t = y.clone()
        for i in range(self.iters):
            # Update x_t
            r_t = (x_t + Atfun(z_t)).view(1, self.H, self.W)
            sigma_hat_t = z_t.norm() / np.sqrt(m)
            x_t = self.denoiser(r_t, std=sigma_hat_t.item())
            _calc_psnr(self, i, x_t, psnr)
            x_t = x_t.view(-1, 1)

            # Update z_t
            z_t = y - Afun(x_t)

            _print(self, i, x_t, sigma_hat_t, psnr)

        output = x_t.view(1, self.H, self.W).cpu()
        return output, psnr, r_t.view(1, self.H, self.W).cpu(), sigma_hat_t.item()

class DAMP(IterativeDenoisingCSRecon):
    """CS solver with Denoising-based Approximate Message Passing."""
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image=None,
                 verbose=False):
        super().__init__(image_shape, denoiser, iters,
                         image, verbose)

    def __call__(self, y, Afun, Atfun, m, n):
        psnr = _setup(self)
        eps = 0.001
        x_t = torch.zeros(n, 1)
        z_t = y.clone()
        for i in range(self.iters):
            # Update x_t
            r_t = (x_t + Atfun(z_t)).view(1, self.H, self.W)
            sigma_hat_t = z_t.norm() / np.sqrt(m)
            x_t = self.denoiser(r_t, std=sigma_hat_t.item()) # shape (1, H, W)
            _calc_psnr(self, i, x_t, psnr)
            x_t = x_t.view(-1, 1)

            # Calculate Divergence of r_t
            noise = generate_noise(r_t.shape, std=1.)
            div = (noise * (self.denoiser(r_t + eps * noise, std=sigma_hat_t.item()) -
                            self.denoiser(r_t, std=sigma_hat_t.item())) / eps).sum()

            # Update z_t
            z_t = y - Afun(x_t) + z_t * (div / m)

            _print(self, i, x_t, sigma_hat_t, psnr)

        output = x_t.view(1, self.H, self.W).cpu()
        return output, psnr, r_t.view(1, self.H, self.W).cpu(), sigma_hat_t.item()

def _setup(self):
    if self.image is not None:
        psnr = torch.zeros(self.iters)
    return psnr

def _calc_psnr(self, i, x_t, psnr):
    if self.image is not None:
        psnr[i] = calc_psnr(x_t, self.image)

def _print(self, i, x_t, sigma_hat_t, psnr):
    if self.verbose:
        if self.image is None:
            print('iter {}, approx. std of effective noise {:.3f}'.format(i, sigma_hat_t))
        else:
            print('iter {}, approx. std of effective noise {:.3f}, PSNR {:.3f}'.format(i, sigma_hat_t, psnr[i]))
            
            
class DAMP_real(IterativeDenoisingCSRecon):
    """CS solver with Denoising-based Approximate Message Passing."""
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image=None,
                 verbose=False):
        super().__init__(image_shape, denoiser, iters,
                         image, verbose)

    def __call__(self, y, Afun, Atfun, m, n, device):
#         psnr = _setup(self)
        eps = 0.001
        n_sqrt = int(np.sqrt(n))
        x_t = torch.zeros(n_sqrt, n_sqrt)
        z_t = y.clone()
        with torch.no_grad():
            
            for i in range(self.iters):
                # Update x_t
                r_t = (x_t + Atfun(z_t)).view(1, self.H, self.W)
    #             r_t = (x_t + Atfun(z_t))
                r_t = torch.real(r_t)
                sigma_hat_t = z_t.norm() / np.sqrt(m)
                x_t = self.denoiser(r_t.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu() # shape (1, H, W)
    #             _calc_psnr(self, i, x_t, psnr)
    #             x_t = x_t.view(-1, 1)

                # Calculate Divergence of r_t
        
        
        
                noise = generate_noise(r_t.shape, std=1.)
                div = (noise * (self.denoiser((r_t + eps * noise).unsqueeze(0).to(device)).cpu() -
                                self.denoiser((r_t).unsqueeze(0).to(device)).cpu()).squeeze(0) / eps).sum()

                # Update z_t
                z_t = y - Afun(x_t) + z_t * (div / m)

                _print(self, i, x_t, sigma_hat_t)

    #         output = x_t.view(1, self.H, self.W).cpu()
            output = x_t.cpu()
        
        return output, r_t.view(1, self.H, self.W).cpu(), sigma_hat_t.item()

def _setup(self):
    if self.image is not None:
        psnr = torch.zeros(self.iters)
    return psnr

def _calc_psnr(self, i, x_t, psnr):
    if self.image is not None:
        psnr[i] = calc_psnr(x_t, self.image)

def _print(self, i, x_t, sigma_hat_t):
    if self.verbose:
        if self.image is None:
            print('iter {}, approx. std of effective noise {:.3f}'.format(i, sigma_hat_t))
        else:
            print('iter {}, approx. std of effective noise {:.3f}, PSNR {:.3f}'.format(i, sigma_hat_t, psnr[i]))
            
            
               
class DAMP_complex(IterativeDenoisingCSRecon):
    """CS solver with Denoising-based Approximate Message Passing."""
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image=None,
                 verbose=False):
        super().__init__(image_shape, denoiser, iters,
                         image, verbose)

    def __call__(self, y, Afun, Atfun, m, n, device,complex_weight):
#         psnr = _setup(self)
        eps = 0.001
        n_sqrt = int(np.sqrt(n))
        x_t = torch.zeros(n_sqrt, n_sqrt)
        z_t = y.clone()
        with torch.no_grad():
            
            for i in range(self.iters):
                
                # Update x_t
                r_t = (x_t + Atfun(z_t)).view(1, self.H, self.W)
                
                sigma_hat_t = z_t.norm() / np.sqrt(m)
            
                x_t = complex_DnCNN_denoise(self.denoiser,r_t,complex_weight,device) # shape ( H, W)

                # Calculate Divergence of r_t
                r_t_jittered = r_t.clone()
                
#                 eta = torch.from_numpy(np.abs(np.amax(r_t_jittered.numpy()))/1000)
                eta = (torch.max(torch.abs(r_t_jittered))/1000)
                
                noise_vec = generate_noise(r_t.shape, std=1.) + 1j * generate_noise(r_t.shape, std=1.)

                r_t_jittered = r_t_jittered + eta * noise_vec
                
                denoised_jittered = complex_DnCNN_denoise(self.denoiser,r_t_jittered,complex_weight,device)
                
                denoised_jittered_np = denoised_jittered.numpy()
                denoised_np = x_t.clone().numpy()
                noise_vec_np = noise_vec.numpy()
                
#                 div = 0.5*((np.dot(np.real(noise_vec_np).reshape(-1),
#                     np.real(denoised_jittered_np - denoised_np).reshape(-1) / eta.numpy())) + (np.dot(np.imag(noise_vec_np).reshape(-1),
#                     np.imag(denoised_jittered_np - denoised_np).reshape(-1) / eta.numpy())))/n
                
                div = np.real(np.dot((noise_vec_np).reshape(-1),(denoised_jittered_np - denoised_np).reshape(-1)/ eta.numpy()))/n
               

                # Update z_t
                z_t = y - Afun(x_t) + z_t * (div / m)

                _print(self, i, x_t, sigma_hat_t)

    #         output = x_t.view(1, self.H, self.W).cpu()
            output = x_t.cpu()
        
        return output, r_t.view(1, self.H, self.W).cpu(), sigma_hat_t.item()
    
    
class DAMP_complex2(IterativeDenoisingCSRecon):
    """CS solver with Denoising-based Approximate Message Passing."""
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image=None,
                 verbose=False):
        super().__init__(image_shape, denoiser, iters,
                         image, verbose)

    def __call__(self, y, Afun, Atfun, m, n, device,complex_weight):
#         psnr = _setup(self)
        eps = 0.001
        n_sqrt = int(np.sqrt(n))
        x_t = torch.zeros(n_sqrt, n_sqrt)
        z_t = y.clone()
        output_mat = torch.zeros(self.iters,n_sqrt, n_sqrt, dtype=torch.cfloat)
        
        with torch.no_grad():
            
            for i in range(self.iters):
                
                beta = 1
                alpha = 1
                no_of_div = 10
                
                # Update x_t
                r_t = (x_t + alpha*np.sqrt(beta)*Atfun(z_t)).view(1, self.H, self.W)
                
                sigma_hat_t = z_t.norm() / np.sqrt(m)
            
                x_t = complex_DnCNN_denoise(self.denoiser,r_t,complex_weight,device) # shape ( H, W)

                # Calculate Divergence of r_t
                

                # Calculate Divergence of r_t
#                 noise = generate_noise(r_t.shape, std=1.)
#                 div = (noise * (self.denoiser(r_t + eps * noise, std=sigma_hat_t.item()) -
#                                 self.denoiser(r_t, std=sigma_hat_t.item())) / eps).sum()

                r_t_jittered = r_t.clone()
                
# #                 eta = torch.from_numpy(np.abs(np.amax(r_t_jittered.numpy()))/1000)
                eta = (torch.max(torch.abs(r_t_jittered))/1000)
#                 eta = eps
                
                noise_vec = generate_noise(r_t.shape, std=1.) + 1j * generate_noise(r_t.shape, std=1.)

                r_t_jittered = r_t_jittered + eta * noise_vec
                
                denoised_jittered = complex_DnCNN_denoise(self.denoiser,r_t_jittered,complex_weight,device)
                
                denoised_jittered_np = denoised_jittered.numpy()
                denoised_np = x_t.clone().numpy()
                noise_vec_np = noise_vec.numpy()

# #                 div = 0.5*((np.dot(np.real(noise_vec_np).reshape(-1),
# #                     np.real(denoised_jittered_np - denoised_np).reshape(-1) / eta.numpy())) + (np.dot(np.imag(noise_vec_np).reshape(-1),
# #                     np.imag(denoised_jittered_np - denoised_np).reshape(-1) / eta.numpy())))/n
                
                div = 0
                for k_foo in range(no_of_div):
                    div_iter = np.real(np.dot((noise_vec_np).reshape(-1),(denoised_jittered_np - denoised_np).reshape(-1)/ eta.numpy()))
                    div = div+div_iter
                    
                div = div/no_of_div

                # Update z_t
#                 z_t = (m/n)*(y - Afun(x_t) + z_t * (div / m))

                z_t = np.sqrt(beta)*(y - Afun(x_t)) + z_t * (div / m)
    
                _print(self, i, x_t, sigma_hat_t)

                output_mat[i,:,:] = x_t.cpu()
    #         output = x_t.view(1, self.H, self.W).cpu()
            output = x_t.cpu()

        
        return output, output_mat, r_t.view(1, self.H, self.W).cpu(), sigma_hat_t.item()
    
    
    
    
def complex_DnCNN_denoise(real_denoiser,r_t,complex_weight,device):
    # r_t : shape ( 1, H, W)
    r_t_real = torch.real(r_t)
    r_t_complex = torch.real(r_t)
    denoised = real_denoiser(r_t_real.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu() + 1j*complex_weight*(r_t_complex.squeeze(0))
    return denoised

def _setup(self):
    if self.image is not None:
        psnr = torch.zeros(self.iters)
    return psnr

def _calc_psnr(self, i, x_t, psnr):
    if self.image is not None:
        psnr[i] = calc_psnr(x_t, self.image)

def _print(self, i, x_t, sigma_hat_t):
    if self.verbose:
        if self.image is None:
            print('iter {}, approx. std of effective noise {:.3f}'.format(i, sigma_hat_t))
        else:
            print('iter {}, approx. std of effective noise {:.3f}, PSNR {:.3f}'.format(i, sigma_hat_t, psnr[i]))
            
            
            