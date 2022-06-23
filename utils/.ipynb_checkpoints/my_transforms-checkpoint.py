from utils import transform as tutil
from utils import general as gutil
import torch
import numpy as np
from numpy import fft
from bm3d import bm3d

from pylops import LinearOperator

## Forward and backward normalized fft and ifft functions 

def forward_masked_fft(image,mask_complement,ret_tensor=False):
    """Masked Normalized FFT of x with low frequency at center

    Args:
        image: numpy image (H,W)
        mask_complement: list of columns to not sample
        ret_tensor: if tensor output required

    Returns:
        fft_of_the_image: has zeros in the locations (columns) where we donot take samples
        
    Note: It is no longer normalized
    """
    X = tutil.fftnc(image,False)
    X[:,mask_complement] = 0
    X = len(image)*X # comment this to normalize
#     print(type(X))
    if ret_tensor:
        return torch.from_numpy(X).to(dtype=torch.complex64)
    else:
        return X.astype(np.complex64)

def forward_masked_fft_radial(image,idx1_complement,idx2_complement,ret_tensor=False):
    """Masked Normalized FFT of x with low frequency at center

    Args:
        image: numpy image (H,W)
        [idx1_complement,idx2_complement]: indecies to not sample
        ret_tensor: if tensor output required

    Returns:
        fft_of_the_image: has zeros in the locations (columns) where we donot take samples
        
    Note: It is no longer normalized
    """
    X = tutil.fftnc(image,False)
    X[idx1_complement,idx2_complement] = 0
#     X = len(image)*X # comment this to normalize
#     print(type(X))
    if ret_tensor:
        return torch.from_numpy(X).to(dtype=torch.complex64)
    else:
        return X.astype(np.complex64)
    
    
def forward_masked_fft_multicoil(image,idx1_complement,idx2_complement,sens_maps,ret_tensor=False):

    stacked_image = np.zeros(sens_maps.shape, dtype=np.complex128)
    
    for i in range(sens_maps.shape[2]):
        stacked_image[:,:,i] = image
    
    sens_image = np.multiply(stacked_image,sens_maps)
    
    X = np.zeros(sens_maps.shape, dtype=np.complex128)
    for i in range(sens_maps.shape[2]):
        X[:,:,i] = tutil.fftnc(sens_image[:,:,i],False)
    X[idx1_complement,idx2_complement,:] = 0
#     X = len(image)*X # comment this to normalize
#     print(type(X))
    if ret_tensor:
        return torch.from_numpy(X).to(dtype=torch.complex64)
    else:
        return X.astype(np.complex64)
    
    
    
def backward_masked_ifft(image_fft,mask_complement,ret_tensor=False):
    """Masked Normalized FFT of x with low frequency at center

    Args:
        image: numpy fft of the image (H,W) (complex)
        mask_complement: list of columns to not sample
        ret_tensor: if tensor output required

    Returns:
        fft_of_the_image: has zeros in the locations (columns) where we donot take samples
    
    Note: It is no longer normalized
    """
    image_fft[:,mask_complement] = 0
    x = tutil.ifftnc(image_fft,False)
    x = len(image_fft)*x # comment this to normalize
    if ret_tensor:
        return torch.from_numpy(x).to(dtype=torch.complex64)
    else:
        return x.astype(np.complex64)
    
def backward_masked_ifft_radial(image_fft,idx1_complement,idx2_complement,ret_tensor=False):
    """Masked Normalized FFT of x with low frequency at center

    Args:
        image: numpy fft of the image (H,W) (complex)
        [idx1_complement,idx2_complement]: indecies to not sample
        ret_tensor: if tensor output required

    Returns:
        fft_of_the_image: has zeros in the locations (columns) where we donot take samples
    
    Note: It is no longer normalized
    """
    image_fft[idx1_complement,idx2_complement] = 0
    x = tutil.ifftnc(image_fft,False)
#     x = len(image_fft)*x # comment this to normalize
    if ret_tensor:
        return torch.from_numpy(x).to(dtype=torch.complex64)
    else:
        return x.astype(np.complex64)
    
    
def backward_masked_ifft_multicoil(image_fft,idx1_complement,idx2_complement,sens_maps,ret_tensor=False):

    image_fft[idx1_complement,idx2_complement,:] = 0
    nc = np.shape(sens_maps)[2]
    
    x = np.zeros((sens_maps.shape[0],sens_maps.shape[0]), dtype=np.complex128)
    
    for coil in range(nc):
        x = x + np.multiply(tutil.ifftnc(image_fft[:,:,coil],False),np.conjugate(sens_maps[:,:,coil]))
    
    if ret_tensor:
        return torch.from_numpy(x).to(dtype=torch.complex64)
    else:
        return x.astype(np.complex64)
    
    
    

def add_noise_to_measurements(y,wvar,mask_complement):
    
    result = y + np.random.normal(0, np.sqrt(wvar), y.shape) # noisy measurement
    result[:,mask_complement] = 0
    
    return result
    

def add_noise_to_complex_measurements(y,wvar,mask_complement):
    
    result_real = y.real + np.random.normal(0, np.sqrt(wvar/2), y.shape)
    result_imag = y.imag + np.random.normal(0, np.sqrt(wvar/2), y.shape)
    
    result = result_real + 1j*result_imag
    result[:,mask_complement] = 0
            
    return result

def add_noise_to_complex_measurements_radial(y,wvar,idx1_complement,idx2_complement):
    
    noise_real = np.random.normal(0, np.sqrt(wvar/2), y.shape)
    noise_imag = np.random.normal(0, np.sqrt(wvar/2), y.shape)
    
    result_real = y.real + noise_real
    result_imag = y.imag + noise_imag
    result = result_real + 1j*result_imag
    result[idx1_complement,idx2_complement] = 0
    
    ## Testing added noise

    noise_real[idx1_complement,idx2_complement] = 0
    noise_imag[idx1_complement,idx2_complement] = 0
    
    pow_1 = np.sum(y.real**2)
    pow_2 = np.sum(y.imag**2)
    pow_3 = np.sum(noise_real**2)
    pow_4 = np.sum(noise_imag**2)
    ratio_snr = np.sqrt(pow_1 + pow_2)/np.sqrt(pow_3 + pow_4)
    SNRdB_test = 20*np.log10(ratio_snr)
    print('SNR in dB for this run:')
    print(SNRdB_test)
    
    ## Done Testing
    
    return result

def add_noise_to_complex_measurements_multicoil(y,wvar,idx1_complement,idx2_complement):
    
    noise_real = np.random.normal(0, np.sqrt(wvar/2), y.shape)
    noise_imag = np.random.normal(0, np.sqrt(wvar/2), y.shape)
    
    result_real = y.real + noise_real
    result_imag = y.imag + noise_imag
    result = result_real + 1j*result_imag
    result[idx1_complement,idx2_complement,:] = 0
    
    ## Testing added noise

    noise_real[idx1_complement,idx2_complement,:] = 0
    noise_imag[idx1_complement,idx2_complement,:] = 0
    
    pow_1 = np.sum(y.real**2)
    pow_2 = np.sum(y.imag**2)
    pow_3 = np.sum(noise_real**2)
    pow_4 = np.sum(noise_imag**2)
    ratio_snr = np.sqrt(pow_1 + pow_2)/np.sqrt(pow_3 + pow_4)
    SNRdB_test = 20*np.log10(ratio_snr)
    print('SNR in dB for this run:')
    print(SNRdB_test)
    
    ## Done Testing
    
    return result






def multiply_tau_subbandwise(wavelet, tau):
    """Modify wavelet subband according to the whitening algorithm.

    Args:
        wavelet (Wavelet): a wavelet.
        tau (list of float): list of tau (standard deviation) in each subband. Size: (level+1)x1

    Returns:
        result (Wavelet): modified wavelet (coloured wavelet?)

    Note:
        Add note here.
    """
    result = wavelet.copy()
    result.coeff[0] = result.coeff[0]*tau[0]
    idx = 1
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            result.coeff[i][j] = result.coeff[i][j]*tau[idx]
        idx += 1
    return result

def multiply_tau_subbandwise_full(wavelet, tau_full):
    """Modify wavelet subband according to the whitening algorithm.

    Args:
        wavelet (Wavelet): a wavelet.
        tau_full (list of float): list of tau (standard deviation) in each subband. Size: (level)x3 + 1

    Returns:
        result (Wavelet): modified wavelet (coloured wavelet?)

    Note:
        Add note here.
    """
    result = wavelet.copy()
    result.coeff[0] = result.coeff[0]*tau_full[0]
    idx = 1
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            result.coeff[i][j] = result.coeff[i][j]*tau_full[idx+j]
        idx += 3
    return result


def get_diag_tau_values(wavelet, tau):
    """Arrange tau so that the entries of the array corresponds to the diagonal matrix's diagonal entries

    Args:
        wavelet (Wavelet): a wavelet.
        tau (list of float): list of tau (standard deviation or inverse standard deviation) in each subband. Size: (level+1)x1

    Returns:
        diag_tau_val (numpy) : array containing diagonal entries
        
    Note:
        Add note here.
    """
    result = wavelet.copy()
    level = len(result.coeff)-1
    n1 = len(result.coeff[0])
    n = n1 + n1*(2**(level)-1)
    
    diag_tau_val = torch.ones(n*n,1)
    
    diag_tau_val[0:len(result.coeff[0])*len(result.coeff[0]),0] = tau[0]
    idx = 1
    start_idx = len(result.coeff[0])*len(result.coeff[0])
    
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            end_idx = start_idx + len(result.coeff[i][j])*len(result.coeff[i][j])
            diag_tau_val[start_idx:end_idx,0] = tau[idx]
            start_idx = end_idx
        idx += 1
#     print(end_idx)
    return diag_tau_val.numpy()

def get_diag_tau_values_full(wavelet, tau_full):
    """Arrange tau_full so that the entries of the array corresponds to the diagonal matrix's diagonal entries

    Args:
        wavelet (Wavelet): a wavelet.
        tau_full (list of float): list of tau (standard deviation or inverse standard deviation) in each subband. Size: (level)x3+1

    Returns:
        diag_tau_val_full (numpy) : array containing diagonal entries
        
    Note:
        Add note here.
    """
    result = wavelet.copy()
    level = len(result.coeff)-1
    n1 = len(result.coeff[0])
    n = n1 + n1*(2**(level)-1)
    
    diag_tau_val_full = torch.ones(n*n,1)
    
    diag_tau_val_full[0:len(result.coeff[0])*len(result.coeff[0]),0] = tau_full[0]
    idx = 1
    start_idx = len(result.coeff[0])*len(result.coeff[0])
    
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            end_idx = start_idx + len(result.coeff[i][j])*len(result.coeff[i][j])
            diag_tau_val_full[start_idx:end_idx,0] = tau_full[idx+j]
            start_idx = end_idx
        idx += 3
#     print(end_idx)
    return diag_tau_val_full.numpy()



def find_least_squares_solution(A_op,y,beta,r,inner_iter):
    
    Ah_y = A_op.H(y)
    
    x = A_op.H(y)
    
    # Approximates the solution of:
    # x = argmin_z 1/(2)||Hz-y||_2^2 + 0.5*beta||z - v||_2^2  or (A^H*A + beta*I)^-1(A^Hy + beta*v)
    
    for j in range(inner_iter):
        b = tutil.add(Ah_y, tutil.mul_subbandwise_scalar(r,beta))
        A_x_est = tutil.add(A_op.H(A_op.A(x)), tutil.mul_subbandwise_scalar(x,beta))
        res = tutil.sub(b,A_x_est)
        a_res = tutil.add(A_op.H(A_op.A(res)), tutil.mul_subbandwise_scalar(res,beta))
        mu_opt = np.abs(tutil.dot_product(res,res))/np.abs(tutil.dot_product(res,a_res))
#         print(mu_opt)
#         mu_opt = torch.mean(res*res)/torch.mean(res*a_res)
#         mu_opt = 0.5
        x = tutil.add(x , tutil.mul_subbandwise_scalar(res,mu_opt))
#         print(type(A_op.A(x)))

    boo = A_op.A(x).numpy() - y            
    resnorm_recov = np.sqrt(np.sum(boo**2))
#         print(resnorm_recov)
#     print('Iter: {0:2d}  '.format(j) + 'Residual Norm: {0:.5f}  '.format(resnorm_recov.real))

    return x

# def find_least_squares_solution(mri,y,inner_iters):
        
#         x = mri.H(y)
        
#         for j in range(inner_iters):
            
#             b = mri.H(y)
#             A_x_est = mri.H(mri.A(x)) 
#             res = b - A_x_est
#             a_res = mri.H(mri.A(res))
#             mu_opt = torch.mean(res*res)/torch.mean(res*a_res)
# #             mu_opt = 1
#             x = x + mu_opt*res
            
#             boo = mri.A(x) - y            
#             resnorm_recov = torch.sqrt(torch.sum(boo**2))
# #             print('Iter: {0:2d}  '.format(j) + 'Residual Norm: {0:.5f}  '.format(resnorm_recov))

#         print('Residual Norm of Least Squares Solution: {0:.18f}  '.format(resnorm_recov))

#         return x

# def find_least_squares_solution_2(mri,y,inner_iters):
        
#         #Dose Basic gradient descent
#         x = mri.H(y)
        
#         for j in range(inner_iters):
            
#             mu_opt = 1
#             x = x + mu_opt*mri.H(y - mri.A(x))
            
#             boo = mri.A(x) - y            
#             resnorm_recov = torch.sqrt(torch.sum(boo**2))
# #             print('Iter: {0:2d}  '.format(j) + 'Residual Norm: {0:.5f}  '.format(resnorm_recov))

#         print('Residual Norm of Least Squares Solution: {0:.18f}  '.format(resnorm_recov))

#         return x


def get_random_p1m1_uniform_wavelet_eta(ref_wavelet,N):
    """get random +-1 uniform wavalet entries

    Args:
        ref_wavelet (Wavelet): a reference wavelet.

    Returns:
        result (Wavelet): a wavelet contaning entries of 2*((np.random.uniform(0, 1,(1,N)) < 0.5)*1 - 0.5)
        
    """
    eta = 2*((np.random.uniform(0, 1, (1,N)) < 0.5)*1 - 0.5)
    result = ref_wavelet.copy()
    start_idx = 0
    end_idx = start_idx + result.coeff[0].size
    result.coeff[0][:,:] = np.reshape(eta[0,start_idx:end_idx],result.coeff[0].shape)
    start_idx = end_idx
    idx = 1
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            end_idx = start_idx + result.coeff[i][j].size
            result.coeff[i][j] = np.reshape(eta[0,start_idx:end_idx],result.coeff[i][j].shape)
            start_idx = end_idx
    return result

def get_MC_p1m1_mat(eta_wav,n,slices):
    """get a 3D matrix of random +-1 uniform entries (given vector) assigned to each each level

    Args:
        eta_wav (Wavelet): wavelet consisting of the random entries.

    Returns:
        result: a 3D matrix with each 2D matrix containing all zeros except at corresponding locations
        
    """

    level = len(eta_wav.coeff) - 1
    result = np.zeros([level+1,n,n])
    
    dummy_zeros = np.zeros([n,n])
    dummy_zeros_wavelet = tutil.pyramid_backward(dummy_zeros, slices)
    
    dummy_zeros_wavelet.coeff[0][:,:] = eta_wav.coeff[0][:,:]
    result[0,:,:] = dummy_zeros_wavelet.pyramid_forward(get_slices=False, to_tensor=False)
    
    for i in range(1, len(eta_wav.coeff)):
        dummy_zeros = np.zeros([n,n])
        dummy_zeros_wavelet = tutil.pyramid_backward(dummy_zeros, slices)
        for j in range(len(eta_wav.coeff[i])):
            dummy_zeros_wavelet.coeff[i][j][:,:] = eta_wav.coeff[i][j][:,:]
        result[i,:,:] = dummy_zeros_wavelet.pyramid_forward(get_slices=False, to_tensor=False)
        
    return result


def get_MC_p1m1_mat_full(eta_wav,n,slices):
    """get a 3D matrix of random +-1 uniform entries (given vector) assigned to each each level

    Args:
        eta_wav (Wavelet): wavelet consisting of the random entries.

    Returns:
        result: a 3D matrix with each 2D matrix containing all zeros except at corresponding locations
        
    """

    level = len(eta_wav.coeff) - 1
    result = np.zeros([level*3+1,n,n])
    
    dummy_zeros = np.zeros([n,n])
    dummy_zeros_wavelet = tutil.pyramid_backward(dummy_zeros, slices)
    
    dummy_zeros_wavelet.coeff[0][:,:] = eta_wav.coeff[0][:,:]
    result[0,:,:] = dummy_zeros_wavelet.pyramid_forward(get_slices=False, to_tensor=False)
    idx = 1
    for i in range(1, len(eta_wav.coeff)):
        for j in range(len(eta_wav.coeff[i])):
            dummy_zeros = np.zeros([n,n])
            dummy_zeros_wavelet = tutil.pyramid_backward(dummy_zeros, slices)
            dummy_zeros_wavelet.coeff[i][j][:,:] = eta_wav.coeff[i][j][:,:]
            result[idx+j,:,:] = dummy_zeros_wavelet.pyramid_forward(get_slices=False, to_tensor=False)
        idx = idx+3
        

    return result



def precon_3Dmat_to_vec(mat,N):
    
    vec = np.zeros((2*N,1),dtype = 'complex_')
    vec[0:N,:] = np.reshape(mat[0,:,:],(N,1))
    vec[N:2*N,:] = np.reshape(mat[1,:,:],(N,1))
    
    return vec

def precon_3Dmat_to_vec_multi_coil(mat1_multicoil,mat2,N,nc):
    vec = np.zeros(((nc + 1)*N,1),dtype = 'complex_')
    start_idx = 0
    for coil in range(nc):
        end_idx = start_idx + N
#         vec[start_idx:end_idx,:] = np.reshape(mat1_multicoil[:,:,coil],(N,1))
        vec[start_idx:end_idx] = np.reshape(mat1_multicoil[:,:,coil],(N,1))
        start_idx = end_idx
#     vec[start_idx:(nc+1)*N,:] = np.reshape(mat2,(N,1))
    vec[start_idx:(nc+1)*N] = np.reshape(mat2,(N,1))
    
    return vec



def precon_vec_to_3Dmat(vec,N):
    
    mat = np.zeros((2,int(np.sqrt(N)),int(np.sqrt(N))),dtype = 'complex_')
    mat[0,:,:] = np.reshape(vec[0:N],(int(np.sqrt(N)),int(np.sqrt(N))))
    mat[1,:,:] = np.reshape(vec[N:2*N],(int(np.sqrt(N)),int(np.sqrt(N))))
    
    return mat

def precon_vec_to_3Dmat_multi_coil(vec,N,nc):
    
    mat1_multicoil = np.zeros((int(np.sqrt(N)),int(np.sqrt(N)),nc),dtype = 'complex_')
    mat2 = np.zeros((int(np.sqrt(N)),int(np.sqrt(N))),dtype = 'complex_')
    
    start_idx = 0
    for coil in range(nc):
        end_idx = start_idx + N
#         mat1_multicoil[:,:,coil] = np.reshape(vec[start_idx:end_idx,:],(int(np.sqrt(N)),int(np.sqrt(N))))
        mat1_multicoil[:,:,coil] = np.reshape(vec[start_idx:end_idx],(int(np.sqrt(N)),int(np.sqrt(N))))
        start_idx = end_idx
        
#     mat2 = np.reshape(vec[start_idx:(nc+1)*N,:],(int(np.sqrt(N)),int(np.sqrt(N))))
    mat2 = np.reshape(vec[start_idx:(nc+1)*N],(int(np.sqrt(N)),int(np.sqrt(N))))
    
    return mat1_multicoil,mat2



def denoise_wavelets_of_the_image_bm3d(image,stds, wavetype, level):
    """Denoises the image in wavelet domain using bm3d

    Args:
        image: noisy numpy image (H,W)
        std: list of stds in different subbands numpy
        wavetype: type of wavelet to use
        level: level of wavelet decomposition

    Returns:
        denoised_image: denoised numpy image (H,w)
        
    Note: This is a new idea to us denoiser in wavelet domain since the wavelet images have good structure and it looks like image
    Note: Tried it and it doesnot denoise well with BM3D.
    Note: image scalling issue exists? 
    """

    wavelet_computed = tutil.forward(image, wavelet=wavetype, level=level)
    result = wavelet_computed.copy()
    result.coeff[0][:,:] = bm3d(wavelet_computed.coeff[0][:,:], stds[0])
    idx = 1
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            result.coeff[i][j][:,:]= bm3d(wavelet_computed.coeff[i][j][:,:], stds[idx])
            idx += 1
    denoised_image = result.inverse(wavelet=wavetype)
#     print(type(denoised_image))
    return denoised_image.numpy()


class preconditioned_A_op_fft_radial_full(LinearOperator):
    
    def __init__(self,idx1_complement,idx2_complement,GAMMA_full,wavetype,level,slices,GAMMA_1_full,sigma_w,N):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.shape = (2*N,N)
        self.GAMMA_full = GAMMA_full
        self.wavetype = wavetype
        self.level = level
        self.slices = slices
        self.GAMMA_1_full = GAMMA_1_full 
        self.sigma_w = sigma_w
        self.N = N
    
    def _matvec(self, x):
        
        y0 = np.reshape(x,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        y00 = tutil.pyramid_backward(y0, self.slices)
        
        y1 = y00.inverse(wavelet=self.wavetype,to_tensor=False)
        y11 = forward_masked_fft_radial(y1,self.idx1_complement,self.idx2_complement,False)

        y2 = multiply_tau_subbandwise_full(y00, self.sigma_w*np.sqrt(self.GAMMA_full*self.GAMMA_1_full))
        y22 = y2.pyramid_forward(get_slices=False, to_tensor=False)
        
        out0 = np.stack((y11,y22))
        out = precon_3Dmat_to_vec(out0,self.N)
        return out
    
    def _rmatvec(self, x):
        
        y00 = precon_vec_to_3Dmat(x,self.N)
        
        y1 = (backward_masked_ifft_radial(y00[0,:,:],self.idx1_complement,self.idx2_complement,False))
        y11 = tutil.forward(y1, wavelet=self.wavetype, level=self.level)
        
        y2 = tutil.pyramid_backward(y00[1,:,:], self.slices)
        y22 = multiply_tau_subbandwise_full(y2, self.sigma_w*np.sqrt(self.GAMMA_full*self.GAMMA_1_full))
        
        y3 = tutil.add(y11, y22)
        
        out0 = y3.pyramid_forward(get_slices=False, to_tensor=False)
        out = np.reshape(out0,(self.N,1))
        
        return out
    
class preconditioned_A_op_fft_radial_Simplified_GEC_full(LinearOperator):
    
    def __init__(self,idx1_complement,idx2_complement,GAMMA_1_full,wavetype,level,slices,sigma_w,N):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.shape = (2*N,N)
        self.GAMMA_1_full = GAMMA_1_full
        self.wavetype = wavetype
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
    
    def _matvec(self, x):
        
        y0 = np.reshape(x,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        y00 = tutil.pyramid_backward(y0, self.slices)
        
        y1 = y00.inverse(wavelet=self.wavetype,to_tensor=False)
        y11 = forward_masked_fft_radial(y1,self.idx1_complement,self.idx2_complement,False)

        y2 = multiply_tau_subbandwise_full(y00, self.sigma_w*np.sqrt(self.GAMMA_1_full))
        y22 = y2.pyramid_forward(get_slices=False, to_tensor=False)
        
        out0 = np.stack((y11,y22))
        out = precon_3Dmat_to_vec(out0,self.N)

        return out
    
    def _rmatvec(self, x):
        
        y00 = precon_vec_to_3Dmat(x,self.N)
        
        y1 = (backward_masked_ifft_radial(y00[0,:,:],self.idx1_complement,self.idx2_complement,False))
        y11 = tutil.forward(y1, wavelet=self.wavetype, level=self.level)
        
        y2 = tutil.pyramid_backward(y00[1,:,:], self.slices)
        y22 = multiply_tau_subbandwise_full(y2, self.sigma_w*np.sqrt(self.GAMMA_1_full))
        
        y3 = tutil.add(y11, y22)
        
        out0 = y3.pyramid_forward(get_slices=False, to_tensor=False)
        out = np.reshape(out0,(self.N,1))
        
        return out
    
    

class preconditioned_multicoil_A_op_fft_Simplified_GEC_full(LinearOperator):
    
    def __init__(self,idx1_complement,idx2_complement,GAMMA_1_full,wavetype,level,slices,sigma_w,N,sens_maps):
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.GAMMA_1_full = GAMMA_1_full
        self.wavetype = wavetype
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
        self.sens_maps = sens_maps
        self.nc = np.shape(self.sens_maps)[2]
        self.shape = ((self.nc+1)*N,N)
    
    def _matvec(self, x):
        
        y0 = np.reshape(x,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        y00 = tutil.pyramid_backward(y0, self.slices)
        
        y1 = y00.inverse(wavelet=self.wavetype,to_tensor=False)
        y11 = forward_masked_fft_multicoil(y1,self.idx1_complement,self.idx2_complement,self.sens_maps,False)

        y2 = multiply_tau_subbandwise_full(y00, self.sigma_w*np.sqrt(self.GAMMA_1_full))
        y22 = y2.pyramid_forward(get_slices=False, to_tensor=False)

        out = precon_3Dmat_to_vec_multi_coil(y11,y22,self.N,self.nc)
        
        return out
    
    def _rmatvec(self, x):
        
        y00_mat1_multicoil,y00_mat2 = precon_vec_to_3Dmat_multi_coil(x,self.N,self.nc)
        
        y1 = backward_masked_ifft_multicoil(y00_mat1_multicoil,self.idx1_complement,self.idx2_complement,self.sens_maps, False)
        y11 = tutil.forward(y1, wavelet=self.wavetype, level=self.level)
        
        y2 = tutil.pyramid_backward(y00_mat2, self.slices)
        y22 = multiply_tau_subbandwise_full(y2, self.sigma_w*np.sqrt(self.GAMMA_1_full))
        
        y3 = tutil.add(y11, y22)
        
        out0 = y3.pyramid_forward(get_slices=False, to_tensor=False)
        out = np.reshape(out0,(self.N,1))
        
        return out