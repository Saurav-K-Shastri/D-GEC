"""D-VDAMP algorithm, MRI measurement, and wavelet denoisers.

    D-VDAMP algorithm
        * dvdamp
        * calc_wavespec

    MRI measurement
        * gen_pdf

    Denoisers
        * calc_MC_divergence
        * ColoredDnCNN_VDAMP
        * BM3D_VDAMP
        * SoftThresholding_VDAMP

    Note:
        The "wavelet" format means list of lists where the position of each value
        belongs to the corresponding wavelet subband. For example, the predicted
        variance tau in dvdamp with level of decomposition = 2 is in the format:

            [A1, [H1, V1, D1], [H2, V2, D2]]

        where each value is the variance corresponding to a subband. Note that
        earlier index (lower number in the example above) means smaller scale.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from numpy import fft
import torch
from bm3d import bm3d
from algorithm.denoiser import ColoredDnCNN
from utils import transform as tutil

import utils.my_transforms as mutil
import scipy 

from pylops import LinearOperator

from scipy.sparse.linalg.interface import aslinearoperator

from utils import *

from utils.noise_model import concatenate_noisy_data_with_a_noise_realization_of_given_stds

from utils.general import load_checkpoint, save_image

import scipy.io
"""VDAMP algorithm"""

def dvdamp(y,
        prob_map,
        mask, 
        var0,
        denoiser,
        image=None,
        iters=30,
        level=4,
        wavetype='haar',
        stop_on_increase=True):
    """Perform VDAMP

    Args:
        y (np.ndarray, (H, W)): MRI measurement.
        prob_map (np.ndarray, (H, W)): sampling probability map.
        mask (np.ndarray, (H, W)): sampling mask generated from prob_map.
        var0 (float): variance of measurement noise.
        image (np.ndarray, (H, W)): Ground truth image.
        iters (int): number of iterations.
        level (int): number of levels for wavelet transform.
        wavetype (str): wavelet type. Refer to pywt for options.
        stop_on_increase (bool): whether to stop D-VDAMP when the predicted MSE increases.

    Returns:
        x_hat (np.ndarray, (H, W)): reconstructed image.
        log (dict): reconstruction log, containing reconstructed image (x_hat),
            wavelet pyramids before (r) and after thresholding (w_hat), and
            true (err) and estimated (tau) RMSE of effective noise for each iteration.
        true_iter (int): actual number of iterations before stopping. This is equal to
            iters if stop_on_increase is False.

    Notes:
        The algorithm is Algorithm 1 described in

            C. A. Metzler and G. Wetzstein, "D-VDAMP: Denoising-Based Approximate Message Passing
            for Compressive MRI," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, 
            Speech and Signal Processing (ICASSP), 2021, pp. 1410-1414, doi: 10.1109/ICASSP39728.2021.9414708.

        This function follows the MATLAB code of VDAMP closely with the soft-thresholding denoiser
        changed to a generic denoiser.

            Charles Millard, Aaron T Hess, Boris Mailhe, and Jared Tanner, 
            “Approximate message passing with a colored aliasing model for variable density 
            fourier sampled images,” arXiv preprint arXiv:2003.02701, 2020.
    """

    # Precompute
    H, W = y.shape
    specX = calc_wavespec(H, level)
    specY = calc_wavespec(W, level)
    Pinv = 1 / prob_map      # Pinv is element-wise inverse
    Pinvm1 = Pinv - 1
    log = {
        'x_hat' : np.zeros((iters, H, W), dtype=complex),
        'r' : np.zeros((iters, H, W), dtype=complex),
        'w_hat' : np.zeros((iters, H, W), dtype=complex),
        'tau' : [None] * iters
    }
    if image is not None:
        w0 = tutil.forward(image, wavelet=wavetype, level=level)
        log = {
            'x_hat' : np.zeros((iters, H, W), dtype=complex),
            'r' : np.zeros((iters, H, W), dtype=complex),
            'w_hat' : np.zeros((iters, H, W), dtype=complex),
            'err' : [None] * iters,
            'tau' : [None] * iters
        }

    # Initialize
    r = tutil.forward(tutil.ifftnc(Pinv * mask * y), wavelet=wavetype, level=level)
    tau_y = mask * Pinv * (Pinvm1 * np.abs(y) ** 2 + var0)
    tau = [None] * (level + 1)
    tau[0] = (specX[:, 0, 0].reshape(1, -1) @ tau_y @ specY[:, 0, 0].reshape(-1, 1)).item()
    for b in range(level):
        tau_b = [None] * 3
        tau_b[0] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 0].reshape(-1, 1)).item()
        tau_b[1] = (specX[:, b, 0].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
        tau_b[2] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
        tau[b + 1] = tau_b
    pred_MSE_prev_iter = np.inf
    true_iters = 0

    _, slices = r.pyramid_forward(get_slices=True)
    log['slices'] = slices

    # Loop
    for it in range(iters):
#         print('iter : ', it)
        # Thresholding
        w_hat, alpha = denoiser(r, tutil.reformat_subband2array(tau))

        # Calculate x_hat
        # x_hat = w_hat.inverse(to_tensor=False)
        x_tilde = w_hat.inverse(to_tensor=False)
        x_hat = x_tilde + tutil.ifftnc(mask * (y - tutil.fftnc(x_tilde)))

        log['r'][it] = r.pyramid_forward(to_tensor=False)
        log['w_hat'][it] = w_hat.pyramid_forward(to_tensor=False)
        log['x_hat'][it] = x_hat
        log['tau'][it] = tau
        if image is not None:
            log['err'][it] = _calc_mse(r, w0)

        if stop_on_increase:
            true_iters += 1
            pred_MSE_this_iter = _calc_pred_mse(tau, level)
            if pred_MSE_this_iter > pred_MSE_prev_iter:
                break
            else:
                pred_MSE_prev_iter = pred_MSE_this_iter
        else:
            true_iters += 1

        # Onsager correction
        w_tilde_coeff = [None] * (level + 1)
        w_tilde_coeff[0] = w_hat.coeff[0] - alpha[0] * r.coeff[0]
        w_div = np.sum(r.coeff[0] * w_tilde_coeff[0]) / (np.sum(w_tilde_coeff[0] ** 2))
        w_tilde_coeff[0] *= w_div
        for b in range(1, level + 1):
            w_tilde_coeff_b = [None] * 3
            for s in range(3):
                w_tilde_coeff_b[s] = w_hat.coeff[b][s] - alpha[b][s] * r.coeff[b][s]
                w_div = np.sum(r.coeff[b][s] * w_tilde_coeff_b[s]) / (np.sum(w_tilde_coeff_b[s] ** 2))
                w_tilde_coeff_b[s] *= w_div
            w_tilde_coeff[b] = w_tilde_coeff_b
        w_tilde = tutil.Wavelet(w_tilde_coeff)

        # Reweighted gradient step
        z = mask * (y - tutil.fftnc(w_tilde.inverse(to_tensor=False)))
        r = tutil.add(w_tilde, tutil.forward(tutil.ifftnc(Pinv * z), wavelet=wavetype, level=level))

        # Noise power re-estimation
        tau_y = mask * Pinv * (Pinvm1 * np.abs(z) ** 2 + var0)
        tau = [None] * (level + 1)
        tau[0] = (specX[:, 0, 0].reshape(1, -1) @ tau_y @ specY[:, 0, 0].reshape(-1, 1)).item()
        for b in range(level):
            tau_b = [None] * 3
            tau_b[0] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 0].reshape(-1, 1)).item()
            tau_b[1] = (specX[:, b, 0].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
            tau_b[2] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
            tau[b + 1] = tau_b

    return x_hat, log, true_iters

def dvdamp2(y,
        prob_map,
        mask, 
        var0,
        denoiser,
        image=None,
        iters=30,
        level=4,
        wavetype='haar',
        stop_on_increase=True):
    """Perform VDAMP

    Args:
        y (np.ndarray, (H, W)): MRI measurement.
        prob_map (np.ndarray, (H, W)): sampling probability map.
        mask (np.ndarray, (H, W)): sampling mask generated from prob_map.
        var0 (float): variance of measurement noise.
        image (np.ndarray, (H, W)): Ground truth image.
        iters (int): number of iterations.
        level (int): number of levels for wavelet transform.
        wavetype (str): wavelet type. Refer to pywt for options.
        stop_on_increase (bool): whether to stop D-VDAMP when the predicted MSE increases.

    Returns:
        x_hat (np.ndarray, (H, W)): reconstructed image.
        log (dict): reconstruction log, containing reconstructed image (x_hat),
            wavelet pyramids before (r) and after thresholding (w_hat), and
            true (err) and estimated (tau) RMSE of effective noise for each iteration.
        true_iter (int): actual number of iterations before stopping. This is equal to
            iters if stop_on_increase is False.

    Notes:
        The algorithm is Algorithm 1 described in

            C. A. Metzler and G. Wetzstein, "D-VDAMP: Denoising-Based Approximate Message Passing
            for Compressive MRI," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, 
            Speech and Signal Processing (ICASSP), 2021, pp. 1410-1414, doi: 10.1109/ICASSP39728.2021.9414708.

        This function follows the MATLAB code of VDAMP closely with the soft-thresholding denoiser
        changed to a generic denoiser.

            Charles Millard, Aaron T Hess, Boris Mailhe, and Jared Tanner, 
            “Approximate message passing with a colored aliasing model for variable density 
            fourier sampled images,” arXiv preprint arXiv:2003.02701, 2020.
    """

    # Precompute
    H, W = y.shape
    specX = calc_wavespec(H, level)
    specY = calc_wavespec(W, level)
    Pinv = 1 / prob_map      # Pinv is element-wise inverse
    Pinvm1 = Pinv - 1
    log = {
        'x_hat' : np.zeros((iters, H, W), dtype=complex),
        'r' : np.zeros((iters, H, W), dtype=complex),
        'w_hat' : np.zeros((iters, H, W), dtype=complex),
        'tau' : [None] * iters
    }
    if image is not None:
        w0 = tutil.forward(image, wavelet=wavetype, level=level)
        log = {
            'x_hat' : np.zeros((iters, H, W), dtype=complex),
            'r' : np.zeros((iters, H, W), dtype=complex),
            'w_hat' : np.zeros((iters, H, W), dtype=complex),
            'err' : [None] * iters,
            'tau' : [None] * iters
        }

    # Initialize
    r = tutil.forward(tutil.ifftnc(Pinv * mask * y), wavelet=wavetype, level=level)
    tau_y = mask * Pinv * (Pinvm1 * np.abs(y) ** 2 + var0)
    tau = [None] * (level + 1)
    tau[0] = (specX[:, 0, 0].reshape(1, -1) @ tau_y @ specY[:, 0, 0].reshape(-1, 1)).item()
    for b in range(level):
        tau_b = [None] * 3
        tau_b[0] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 0].reshape(-1, 1)).item()
        tau_b[1] = (specX[:, b, 0].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
        tau_b[2] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
        tau[b + 1] = tau_b
    pred_MSE_prev_iter = np.inf
    true_iters = 0

    _, slices = r.pyramid_forward(get_slices=True)
    log['slices'] = slices
    
    x_hat_mat = np.zeros((iters, H, W), dtype=complex)
    # Loop
    for it in range(iters):
#         print('iter : ', it)
        # Thresholding
        w_hat, alpha = denoiser(r, tutil.reformat_subband2array(tau))

        # Calculate x_hat
        # x_hat = w_hat.inverse(to_tensor=False)
        x_tilde = w_hat.inverse(to_tensor=False)
        x_hat = x_tilde + tutil.ifftnc(mask * (y - tutil.fftnc(x_tilde)))

        x_hat_mat[it,:,:] = x_hat
        
        log['r'][it] = r.pyramid_forward(to_tensor=False)
        log['w_hat'][it] = w_hat.pyramid_forward(to_tensor=False)
        log['x_hat'][it] = x_hat
        log['tau'][it] = tau
        if image is not None:
            log['err'][it] = _calc_mse(r, w0)

        if stop_on_increase:
            true_iters += 1
            pred_MSE_this_iter = _calc_pred_mse(tau, level)
            if pred_MSE_this_iter > pred_MSE_prev_iter:
                break
            else:
                pred_MSE_prev_iter = pred_MSE_this_iter
        else:
            true_iters += 1

        # Onsager correction
        w_tilde_coeff = [None] * (level + 1)
        w_tilde_coeff[0] = w_hat.coeff[0] - alpha[0] * r.coeff[0]
        w_div = np.sum(r.coeff[0] * w_tilde_coeff[0]) / (np.sum(w_tilde_coeff[0] ** 2))
        w_tilde_coeff[0] *= w_div
        for b in range(1, level + 1):
            w_tilde_coeff_b = [None] * 3
            for s in range(3):
                w_tilde_coeff_b[s] = w_hat.coeff[b][s] - alpha[b][s] * r.coeff[b][s]
                w_div = np.sum(r.coeff[b][s] * w_tilde_coeff_b[s]) / (np.sum(w_tilde_coeff_b[s] ** 2))
                w_tilde_coeff_b[s] *= w_div
            w_tilde_coeff[b] = w_tilde_coeff_b
        w_tilde = tutil.Wavelet(w_tilde_coeff)

        # Reweighted gradient step
        z = mask * (y - tutil.fftnc(w_tilde.inverse(to_tensor=False)))
        r = tutil.add(w_tilde, tutil.forward(tutil.ifftnc(Pinv * z), wavelet=wavetype, level=level))

        # Noise power re-estimation
        tau_y = mask * Pinv * (Pinvm1 * np.abs(z) ** 2 + var0)
        tau = [None] * (level + 1)
        tau[0] = (specX[:, 0, 0].reshape(1, -1) @ tau_y @ specY[:, 0, 0].reshape(-1, 1)).item()
        for b in range(level):
            tau_b = [None] * 3
            tau_b[0] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 0].reshape(-1, 1)).item()
            tau_b[1] = (specX[:, b, 0].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
            tau_b[2] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
            tau[b + 1] = tau_b

    return x_hat,x_hat_mat, log, true_iters




def _calc_mse(test, ref):
    """Calculate band-wise mean squared error (MSE).

    Args:
        test (util.transform.Wavelet): noisy (test) wavelet.
        ref (util.transform.Wavelet): ground truth (reference) wavelet.

    Returns:
        mse (list): list of MSE in the "wavelet" format.
    """
    mse = [None] * (test.get_bands() + 1)
    mse[0] = np.mean(np.abs(test.coeff[0] - ref.coeff[0]) ** 2)
    for b in range(1, test.get_bands() + 1):
        mse_band = [None] * 3
        for s in range(3):
            mse_band[s] = np.mean(np.abs(test.coeff[b][s] - ref.coeff[b][s]) ** 2)
        mse[b] = mse_band
    return mse

def _calc_pred_mse(tau, level):
    """Calculate a scaled predicted MSE based on 
    the predicted noise variance and the level of wavelet decomposition.

    Args:
        tau (list): predicted noise variance in each subband in the "wavelet" format.

    Returns:
        pred_mse (float): scaled predicted MSE.

    Note:
        The predicted MSE is used to determine whether to stop D-VDAMP early.
    """
    pred_mse = 0
    pred_mse += tau[0] * (4 ** (-level))
    for b in range(level):
        weight = 4 ** (b - level)
        for s in range(3):
            pred_mse += tau[b + 1][s] * weight
    return pred_mse

def calc_wavespec(numsamples, level, wavetype='haar', ret_tensor=False):
    """Calculate power spectrum of wavelet decomposition kernels.

    Returns:
        spec: (numsamples, level, [lowpass, highpass]) power spectrum.
            In axis=1 (level), higher indices mean larger scales.
    """

    wavelet = tutil.Wavelet_bank(wavetype)
    spec = np.zeros([numsamples, level, 2])

    # Zero-pad decomposition filters
    L = np.zeros(numsamples)
    L[0:len(wavelet.dec_lo)] = wavelet.dec_lo
    H = np.zeros(numsamples)
    H[0:len(wavelet.dec_hi)] = wavelet.dec_hi

    # Spectrum of the largest scale
    spec[:, 0, 0] = np.abs(fft.fft(L)) ** 2
    spec[:, 0, 1] = np.abs(fft.fft(H)) ** 2

    # Spectrum of other scales
    numblock = 1
    for s in range(1, level):
        numblock *= 2
        spec[:, s, 0] = spec[:, s - 1, 0] * \
            np.transpose(spec[::numblock, 0, 0].reshape(-1, 1) @ np.ones([1, numblock])).reshape(-1)
        spec[:, s, 1] = spec[:, s - 1, 0] * \
            np.transpose(spec[::numblock, 0, 1].reshape(-1, 1) @ np.ones([1, numblock])).reshape(-1)
    spec = fft.fftshift(spec, axes=0) / numsamples

    if ret_tensor:
        return torch.from_numpy(spec).flip(1).to(torch.float32)
    else:
        return np.flip(spec, axis=1)

"""Functions for MRI sampling simulation"""

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

"""Denoising functions for VDAMP"""

def calc_MC_divergence(denoiser, denoised, wavelet, variances):
    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach.

    Note:
        See section 2.1 of Metzler and Wetzstein 2020, "D-VDAMP: Denoising-Based Approximate Message Passing
        for Compressive MRI" for the equation.
    """
    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    wavelet_jittered = wavelet.copy()
    eta = np.abs(wavelet_jittered.coeff[0].max()) / 1000.
    noise_vec = np.random.randn(*wavelet_jittered.coeff[0].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[0].shape)
    wavelet_jittered.coeff[0] += eta * noise_vec
    denoised_jittered = denoiser(wavelet_jittered, variances)
    alpha[0] = 0.5 * (
                1. / wavelet_jittered.coeff[0].size * np.dot(np.real(noise_vec).reshape(-1),
                    np.real(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta) + # real part
                1. / wavelet_jittered.coeff[0].size * np.dot(np.imag(noise_vec).reshape(-1),
                    np.imag(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta)) # img part
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            wavelet_jittered = wavelet.copy()
            eta = np.abs(wavelet_jittered.coeff[s + 1][b].max()) / 1000.
            noise_vec = np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)
            
            wavelet_jittered.coeff[s + 1][b] += eta * noise_vec

            denoised_jittered = denoiser(wavelet_jittered, variances)
            alpha[s + 1][b] = 0.5 * (
                    1. / wavelet_jittered.coeff[s + 1][b].size * np.dot(np.real(noise_vec).reshape(-1),
                        np.real(denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta) +  # real part
                    1. / wavelet_jittered.coeff[s + 1][b].size * np.dot(np.imag(noise_vec).reshape(-1),
                        np.imag(denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta)) # img part
    return alpha


def calc_MC_divergence_real(denoiser, denoised, wavelet, variances):
    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach.

    Note:
        See section 2.1 of Metzler and Wetzstein 2020, "D-VDAMP: Denoising-Based Approximate Message Passing
        for Compressive MRI" for the equation.
    """
    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    wavelet_jittered = wavelet.copy()
    eta = np.abs(wavelet_jittered.coeff[0].max()) / 1000.
    
    noise_vec = np.random.randn(*wavelet_jittered.coeff[0].shape)
    
    wavelet_jittered.coeff[0] += eta * noise_vec
    
    denoised_jittered = denoiser(wavelet_jittered, variances)
    
    alpha[0] = 1. / wavelet_jittered.coeff[0].size * np.dot(np.real(noise_vec).reshape(-1), np.real(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta)
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            wavelet_jittered = wavelet.copy()
            eta = np.abs(wavelet_jittered.coeff[s + 1][b].max()) / 1000.
            noise_vec = np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)
            wavelet_jittered.coeff[s + 1][b] += eta * noise_vec

            denoised_jittered = denoiser(wavelet_jittered, variances)
            alpha[s + 1][b] = 1. / wavelet_jittered.coeff[s + 1][b].size * np.dot(np.real(noise_vec).reshape(-1), np.real(denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta)
            
    return alpha




def calc_MC_divergence_2(denoiser, denoised, wavelet, variances):
#     print('hi')
    # modified by Saurav
    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach.

    Note:
        See section 2.1 of Metzler and Wetzstein 2020, "D-VDAMP: Denoising-Based Approximate Message Passing
        for Compressive MRI" for the equation.
    """
    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    wavelet_jittered = wavelet.copy()
    eta1 = np.abs(wavelet_jittered.coeff[0].max()) / 1000.
    eta2 = np.mean(np.sqrt(variances))
    eta = np.max([eta1,eta2])
    eps = 2.22e-16 # added by Saurav
    eta = eta + eps # added by Saurav
    noise_vec = np.random.randn(*wavelet_jittered.coeff[0].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[0].shape)
    wavelet_jittered.coeff[0] += eta * noise_vec
    denoised_jittered = denoiser(wavelet_jittered, variances)
    
    alpha[0] = 0.5 * (
                1. / wavelet_jittered.coeff[0].size * np.dot(np.real(noise_vec).reshape(-1),
                    np.real(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta) + # real part
                1. / wavelet_jittered.coeff[0].size * np.dot(np.imag(noise_vec).reshape(-1),
                    np.imag(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta)) # img part
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            wavelet_jittered = wavelet.copy()
            eta1 = np.abs(wavelet_jittered.coeff[s + 1][b].max()) / 1000.
            eta = np.max([eta1,eta2])
            eps = 2.22e-16 # added by Saurav
            eta = eta + eps # added by Saurav
            noise_vec = np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)
            
            wavelet_jittered.coeff[s + 1][b] += eta * noise_vec

            denoised_jittered = denoiser(wavelet_jittered, variances)
            alpha[s + 1][b] = 0.5 * (
                    1. / wavelet_jittered.coeff[s + 1][b].size * np.dot(np.real(noise_vec).reshape(-1),
                        np.real(denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta) +  # real part
                    1. / wavelet_jittered.coeff[s + 1][b].size * np.dot(np.imag(noise_vec).reshape(-1),
                        np.imag(denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta)) # img part
    return alpha

def calc_MC_divergence_3_complex(denoiser, denoised, wavelet, variances):
#     print('complex divergence computation +-1')
    # modified by Saurav
    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach.

    Note:
        See section 2.1 of Metzler and Wetzstein 2020, "D-VDAMP: Denoising-Based Approximate Message Passing
        for Compressive MRI" for the equation.
    """
    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    wavelet_jittered = wavelet.copy()
    eta1 = np.abs(wavelet_jittered.coeff[0].max()) / 1000.
    eta2 = np.mean(np.sqrt(variances))
    eta = np.max([eta1,eta2])
    eps = 2.22e-16 # added by Saurav
    eta = eta + eps # added by Saurav
#     noise_vec = np.sqrt(1/2)*(np.random.randn(*wavelet_jittered.coeff[0].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[0].shape))
    noise_vec = np.sqrt(1/2)*(np.sign(np.random.randn(*wavelet_jittered.coeff[0].shape)) + 1j * np.sign(np.random.randn(*wavelet_jittered.coeff[0].shape)))
    
    wavelet_jittered.coeff[0] += eta * noise_vec
    
    denoised_jittered = denoiser(wavelet_jittered, variances)
    
    alpha[0] = 1. / wavelet_jittered.coeff[0].size * np.real(np.dot( np.conj(noise_vec).reshape(-1),(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta))
    
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            wavelet_jittered = wavelet.copy()
            eta1 = np.abs(wavelet_jittered.coeff[s + 1][b].max()) / 1000.
            eta = np.max([eta1,eta2])
            eps = 2.22e-16 # added by Saurav
            eta = eta + eps # added by Saurav
            
#             noise_vec = np.sqrt(1/2)*(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape))
            noise_vec = np.sqrt(1/2)*(np.sign(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)) + 1j * np.sign(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)))

            
            wavelet_jittered.coeff[s + 1][b] += eta * noise_vec

            denoised_jittered = denoiser(wavelet_jittered, variances)
            alpha[s + 1][b] = 1. / wavelet_jittered.coeff[s + 1][b].size * np.real(np.dot(np.conj(noise_vec).reshape(-1), (denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta)) 
                    
    return alpha


def calc_MC_LMSE_divergence_complex(denoiser, denoised, wavelet, variances, y_dummy, wavetype):
#     print('complex divergence computation +-1 for LMSE!')
    # modified by Saurav

    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    
    probing_vec = np.zeros(y_dummy.shape,dtype=np.complex128) 

    wavelet_jittered = tutil.forward(probing_vec, wavelet=wavetype, level=level)

#     wavelet_jittered = wavelet.copy()
    
    noise_vec = np.sqrt(1/2)*(np.sign(np.random.randn(*wavelet_jittered.coeff[0].shape)) + 1j * np.sign(np.random.randn(*wavelet_jittered.coeff[0].shape)))
    
    wavelet_jittered.coeff[0] += noise_vec
    
    zero_y = y_dummy*0
    
    denoised_jittered = denoiser(wavelet_jittered, variances, zero_y)
    
    alpha[0] = 1. / wavelet_jittered.coeff[0].size * np.real(np.dot( np.conj(noise_vec).reshape(-1),(denoised_jittered.coeff[0]).reshape(-1)))
    
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            
            probing_vec = np.zeros(y_dummy.shape,dtype=np.complex128)
            wavelet_jittered = tutil.forward(probing_vec, wavelet=wavetype, level=level)

            noise_vec = np.sqrt(1/2)*(np.sign(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)) + 1j * np.sign(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)))

            
            wavelet_jittered.coeff[s + 1][b] += noise_vec

            denoised_jittered = denoiser(wavelet_jittered, variances, zero_y)
            
            alpha[s + 1][b] = 1. / wavelet_jittered.coeff[s + 1][b].size * np.real(np.dot(np.conj(noise_vec).reshape(-1), (denoised_jittered.coeff[s + 1][b]).reshape(-1))) 
                    
    return alpha





def calc_MC_divergence_2_real(denoiser, denoised, wavelet, variances):
    print('real valued divergence +-1')
    # modified by Saurav
    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach.

    Note:
        See section 2.1 of Metzler and Wetzstein 2020, "D-VDAMP: Denoising-Based Approximate Message Passing
        for Compressive MRI" for the equation.
    """
    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    wavelet_jittered = wavelet.copy()
    eta1 = np.abs(wavelet_jittered.coeff[0].max()) / 1000.
    eta2 = np.mean(np.sqrt(variances))
    eta = np.max([eta1,eta2])
    eps = 2.22e-16 # added by Saurav
    eta = eta + eps # added by Saurav
    
#     noise_vec = np.random.randn(*wavelet_jittered.coeff[0].shape)
    
    noise_vec = np.sign(np.random.randn(*wavelet_jittered.coeff[0].shape))
    
    
    wavelet_jittered.coeff[0] += eta * noise_vec
    denoised_jittered = denoiser(wavelet_jittered, variances)
    
    alpha[0] = 1. / wavelet_jittered.coeff[0].size * np.real(np.dot(noise_vec.reshape(-1), (denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta))
    
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            wavelet_jittered = wavelet.copy()
            eta1 = np.abs(wavelet_jittered.coeff[s + 1][b].max()) / 1000.
            eta = np.max([eta1,eta2])
            eps = 2.22e-16 # added by Saurav
            eta = eta + eps # added by Saurav
            
#             noise_vec = np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)
            noise_vec = np.sign(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape))
    
            wavelet_jittered.coeff[s + 1][b] += eta * noise_vec

            denoised_jittered = denoiser(wavelet_jittered, variances)
            alpha[s + 1][b] = 1. / wavelet_jittered.coeff[s + 1][b].size * np.real(np.dot(noise_vec.reshape(-1), (denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta))
            
    return alpha








class ColoredDnCNN_VDAMP:
    """Wrapper of algorithm.denoiser.ColoredDnCNN for using with D-VDAMP.
    
    Note:
        In D-VDAMP, the noisy wavelets have complex coefficients.

        For real ground truths (model channel = 1), we appply the denoiser to
        only the real part and scale the imaginary part by 0.1.

        For complex ground truths (model channel = 2), we pass a tensor where
        the first channel is the real part and the second channel is the imaginary part
        to the model. However, this feature is not currently supported.
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = 1, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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
        self.models = self._load_models(modeldir, num_layers, std_channels)
        self.std_pool_func = std_pool_func
        self.beta_tune = beta_tune
        self.verbose = verbose
        

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
#         variances *= gamma
        variances *= self.beta_tune
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = torch.from_numpy(variances).sqrt().to(device=self.device, dtype=torch.float32)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('ColoredDnCNN_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = wavelet.inverse().unsqueeze(0)
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            denoised_real = self.models[select](noisy_real, stds).cpu()
            denoised_imag = noisy_imag * 0.1
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet = tutil.forward(denoised_image[0], wavelet=self.wavetype, level=level)
        elif self.channels == 2:
            noisy_image = torch.vstack([noisy_image.real, noisy_image.imag])
            noisy_image = noisy_image.to(device=self.device, dtype=torch.float32)
            denoised_image = self.models[select](noisy_image, stds).cpu()
            denoised_image = denoised_image[0] + 1j * denoised_image[1]
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

    def _load_models(self, modeldirs, num_layers, std_channels):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = ColoredDnCNN(channels=self.channels, num_layers=num_layers, std_channels=std_channels)
            load_checkpoint(modeldir, model, None, device=self.device)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models
    
    
class ColoredDnCNN_VDAMP_real:
    """Wrapper of algorithm.denoiser.ColoredDnCNN for using with D-VDAMP.
    
    Note:
        In D-VDAMP, the noisy wavelets have complex coefficients.

        For real ground truths (model channel = 1), we appply the denoiser to
        only the real part and scale the imaginary part by 0.1.

        For complex ground truths (model channel = 2), we pass a tensor where
        the first channel is the real part and the second channel is the imaginary part
        to the model. However, this feature is not currently supported.
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = 1, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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
        self.models = self._load_models(modeldir, num_layers, std_channels)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.beta_tune = beta_tune

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_2_real(self._denoise, denoised, wavelet, variances)
#             alpha = calc_MC_divergence_3_complex(self._denoise, denoised, wavelet, variances)
#             print('Note: using multiple iterations for computing divergence!')
#             alpha = calc_MC_divergence_multiple_iteration_real(self._denoise, denoised, wavelet, variances)


            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = torch.from_numpy(variances).sqrt().to(device=self.device, dtype=torch.float32)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('ColoredDnCNN_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = wavelet.inverse().unsqueeze(0)
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            denoised_real = self.models[select](noisy_real, stds).cpu()
            denoised_image = denoised_real
            denoised_wavelet = tutil.forward(denoised_image[0], wavelet=self.wavetype, level=level)
        elif self.channels == 2:
            noisy_image = torch.vstack([noisy_image.real, noisy_image.imag])
            noisy_image = noisy_image.to(device=self.device, dtype=torch.float32)
            denoised_image = self.models[select](noisy_image, stds).cpu()
            denoised_image = denoised_image[0]
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

    def _load_models(self, modeldirs, num_layers, std_channels):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = ColoredDnCNN(channels=self.channels, num_layers=num_layers, std_channels=std_channels)
            load_checkpoint(modeldir, model, None, device=self.device)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    
class ColoredDnCNN_VDAMP_complex:
    """Wrapper of algorithm.denoiser.ColoredDnCNN for using with D-VDAMP.
    
    Note:
        In D-VDAMP, the noisy wavelets have complex coefficients.

        For real ground truths (model channel = 1), we appply the denoiser to
        only the real part and scale the imaginary part by self.complex_weight.

        For complex ground truths (model channel = 2), we pass a tensor where
        the first channel is the real part and the second channel is the imaginary part
        to the model. However, this feature is not currently supported.
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = 1, complex_weight = 0.1, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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
        self.models = self._load_models(modeldir, num_layers, std_channels)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.beta_tune = beta_tune
        self.complex_weight = complex_weight

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            
            alpha = calc_MC_divergence_3_complex(self._denoise, denoised, wavelet, variances)
            
            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = torch.from_numpy(variances).sqrt().to(device=self.device, dtype=torch.float32)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('ColoredDnCNN_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = wavelet.inverse().unsqueeze(0)
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            denoised_real = self.models[select](noisy_real, stds).cpu()
            denoised_imag = noisy_imag * self.complex_weight
            denoised_image = denoised_real + 1j * denoised_imag
#             print(denoised_image.shape)
#             print('normalizing')
#             denoised_image[0] = denoised_image[0]/torch.max(torch.abs(denoised_image[0]))
            denoised_wavelet = tutil.forward(denoised_image[0], wavelet=self.wavetype, level=level)
        elif self.channels == 2:
            noisy_image = torch.vstack([noisy_image.real, noisy_image.imag])
            noisy_image = noisy_image.to(device=self.device, dtype=torch.float32)
            denoised_image = self.models[select](noisy_image, stds).cpu()
            denoised_image = denoised_image[0] + 1j * denoised_image[1]
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

    def _load_models(self, modeldirs, num_layers, std_channels):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = ColoredDnCNN(channels=self.channels, num_layers=num_layers, std_channels=std_channels)
            load_checkpoint(modeldir, model, None, device=self.device)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    
class ColoredDnCNN_VDAMP_complex_vector_beta:
    """Wrapper of algorithm.denoiser.ColoredDnCNN for using with D-VDAMP.
    
    Note:
        In D-VDAMP, the noisy wavelets have complex coefficients.

        For real ground truths (model channel = 1), we appply the denoiser to
        only the real part and scale the imaginary part by self.complex_weight.

        For complex ground truths (model channel = 2), we pass a tensor where
        the first channel is the real part and the second channel is the imaginary part
        to the model. However, this feature is not currently supported.
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune_vector = np.ones(13,), complex_weight = 0.1, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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
        self.models = self._load_models(modeldir, num_layers, std_channels)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.beta_tune_vector = beta_tune_vector
        self.complex_weight = complex_weight

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
        variances *= self.beta_tune_vector
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            
            alpha = calc_MC_divergence_3_complex(self._denoise, denoised, wavelet, variances)
            
            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = torch.from_numpy(variances).sqrt().to(device=self.device, dtype=torch.float32)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('ColoredDnCNN_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = wavelet.inverse().unsqueeze(0)
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            denoised_real = self.models[select](noisy_real, stds).cpu()
            denoised_imag = noisy_imag * self.complex_weight
            denoised_image = denoised_real + 1j * denoised_imag
#             print(denoised_image.shape)
#             print('normalizing')
#             denoised_image[0] = denoised_image[0]/torch.max(torch.abs(denoised_image[0]))
            denoised_wavelet = tutil.forward(denoised_image[0], wavelet=self.wavetype, level=level)
        elif self.channels == 2:
            noisy_image = torch.vstack([noisy_image.real, noisy_image.imag])
            noisy_image = noisy_image.to(device=self.device, dtype=torch.float32)
            denoised_image = self.models[select](noisy_image, stds).cpu()
            denoised_image = denoised_image[0] + 1j * denoised_image[1]
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

    def _load_models(self, modeldirs, num_layers, std_channels):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = ColoredDnCNN(channels=self.channels, num_layers=num_layers, std_channels=std_channels)
            load_checkpoint(modeldir, model, None, device=self.device)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    
    
    

class BM3D_VDAMP:
    """Wrapper of BM3D for using with D-VDAMP."""
    def __init__(self, channels, wavetype='haar', std_pool_func=np.max):
        """Initialize BM3D_VDAMP

        Args:
            channels (int): number of channels to apply BM3D.
                If channels == 1, apply BM3D to the real part and scale the imaginary part by 0.1.
                If channels == 2, apply BM3D to both real and imaginary parts seperately.
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.channels = channels
        self.std_pool_func = std_pool_func
        self.wavetype = wavetype

    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()
        std_pooled = self.std_pool_func(np.sqrt(variances))
        noisy_image = wavelet.inverse()
        if self.channels == 1:
            noisy_real = noisy_image.real
            noisy_imag = noisy_image.imag
            denoised_real = torch.Tensor(bm3d(noisy_real, std_pooled))
            denoised_imag = noisy_imag * 0.1
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        elif self.channels == 2:
            noisy_real = noisy_image.real
            noisy_imag = noisy_image.imag
            denoised_real = torch.Tensor(bm3d(noisy_real, std_pooled))
            denoised_imag = torch.Tensor(bm3d(noisy_imag, std_pooled))
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

class SoftThresholding_VDAMP:
    """Wrapper of soft-thresholding for using with D-VDAMP.
    
    Note:
        With soft-thresholding as the denoiser, the D-VDAMP algorithm becomes the base VDAMP.

            Charles Millard, Aaron T Hess, Boris Mailhe, and Jared Tanner, 
            “Approximate message passing with a colored aliasing model for variable density 
            fourier sampled images,” arXiv preprint arXiv:2003.02701, 2020.

    """
    def __init__(self, MC_divergence, debug=False):
        """Initialize SoftThresholding_VDAMP

        Args:
            MC_divergence (bool): whether to calculate the divergence with the 
                Monte Carlo approach (True) or analytically (False).
            debug (bool): whether to calculate the analytical divergence as well
                for comparison when MC_divergence is True.
        """
        self.MC_divergence = MC_divergence
        self.debug = debug

    def __call__(self, wavelet, variances, calc_divergence=True):
        variances = tutil.reformat_array2subband(variances)
        denoised, df, _ = multiscaleSureSoft(wavelet, variances)
        if calc_divergence:
            if self.MC_divergence:
                alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
                if self.debug:
                    alpha_ana = self._calc_ana_div(df)
                    alpha_array = tutil.reformat_subband2array(alpha)
                    alpha_ana_array = tutil.reformat_subband2array(alpha_ana)
                    error = ((alpha_array - alpha_ana_array) / alpha_ana_array * 100).mean()
                    print('MC div error: {} %'.format(error))
            else:
                alpha = self._calc_ana_div(df)
            return denoised, alpha
        else:
            return denoised

    # wrapper for calc_MC_divergence
    def _denoise(self, wavelet, variances):
        denoised, _, _ = multiscaleSureSoft(wavelet, variances)
        return denoised

    def _calc_ana_div(self, df):
        level = len(df) - 1
        alpha = [None] * (level + 1)
        alpha[0] = np.mean(df[0]) / 2
        for b in range(level):
            alpha_b = [None] * 3
            for s in range(3):
                alpha_b[s] = np.mean(df[b + 1][s]) / 2
            alpha[b + 1] = alpha_b
        return alpha

"""Base implementation of soft thresholding"""

def complexSoft(wavelet_coeff, threshold):
    """Perform soft-thresholding on a wavelet subband given a threshold.

    Args:
        wavelet_coeff (np.ndarray): array of wavelet coefficients.
        threshold (float): the threshold.

    Returns:
        thresholded_coeff (np.ndarray): the thresholded wavelet coefficients.
        df (np.ndarray): degree of freedom, shape like thresholded_coeff

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """
    ones = np.ones_like(wavelet_coeff, dtype=float)
    mag = np.abs(wavelet_coeff)
    gdual = np.minimum(threshold / mag, ones)
    thresholded_coeff = wavelet_coeff * (ones - gdual)
    df = 2 * ones - (2 * ones - (gdual < 1)) * gdual
    return thresholded_coeff, df

def sureSoft(wavelet_coeff, var):
    """
    Perform soft-thresholding on a wavelet subband using optimal threshold estimated with SURE.

    Args:
        wavelet_coeff (np.ndarray): array of wavelet coefficients.
        var (float): variance of wavelet_coeff.

    Returns:
        thresholded_coeff (np.ndarray): the thresholded wavelet coefficients.
        df (np.ndarray): degree of freedom, shape like thresholded_coeff
        threshold (float): the threshold used.

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """

    mag_flat = np.abs(wavelet_coeff.reshape(-1))
    index = np.flipud(np.argsort(np.abs(mag_flat)))
    lamb = mag_flat[index]

    V = var * np.ones_like(mag_flat)
    V = V[index]

    z0 = np.ones_like(lamb)

    SURE_inf = np.flipud(np.cumsum(np.flipud(lamb ** 2)))
    SURE_sup = np.cumsum(z0) * (lamb ** 2) - lamb * np.cumsum(V / lamb) + 2 * np.cumsum(V)
    SURE = SURE_inf + SURE_sup - np.sum(V)

    idx = np.argmin(SURE)
    thresholded_coeff, df = complexSoft(wavelet_coeff, lamb[idx])
    threshold = lamb[idx]

    return thresholded_coeff, df, threshold

def multiscaleComplexSoft(wavelet, variances, lambdas):
    """
    Perform soft-thresholding on wavelet coefficients given sparse weighings.

    Args:
        wavelet (util.transform.Wavelet): the target wavelet.
        variances (list): estimated variances of the bands.
        lambdas (list): sparse weighings.

    Returns:
        thresholded_wavelet (util.transform.Wavelet): the thresholded wavelet.
        df (list): degrees of freedom of bands.

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """
    
    scales = wavelet.get_bands()
    thresholded_wavelet = wavelet.copy()
    df = [None] * (scales + 1)

    thresholded_wavelet.coeff[0], df[0] = complexSoft(wavelet.coeff[0], variances[0] * lambdas[0])
    for i in range(1, scales + 1):
        df_subband = [None] * 3
        for j in range(3):
            thresholded_wavelet.coeff[i][j], df_subband[j] = complexSoft(wavelet.coeff[i][j], variances[i][j] * lambdas[i][j])
        df[i] = df_subband

    return thresholded_wavelet, df

def multiscaleSureSoft(wavelet, variances):
    """
    Perform soft-thresholding on wavelet coefficients using optimal thresholds estimated with SURE.

    Args:
        wavelet (util.transform.Wavelet): the target wavelet.
        variances (list): estimated variances of the bands.

    Returns:
        thresholded_wavelet (util.transform.Wavelet): the thresholded wavelet.
        df (list): degrees of freedom of bands.
        thres (list): used thresholds for bands.

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """
    
    scales = wavelet.get_bands()
    thresholded_wavelet = wavelet.copy()
    df = [None] * (scales + 1)
    thres = [None] * (scales + 1)

    thresholded_wavelet.coeff[0], df[0], thres[0] = sureSoft(wavelet.coeff[0], variances[0])
    for i in range(1, scales + 1):
        df_subband = [None] * 3
        thres_subband = [None] * 3
        for j in range(3):    
            thresholded_wavelet.coeff[i][j], df_subband[j], thres_subband[j] = sureSoft(wavelet.coeff[i][j], variances[i][j])
        df[i] = df_subband
        thres[i] = thres_subband

    return thresholded_wavelet, df, thres


class BM3D_VDAMP_Whitened:
    """Wrapper of BM3D for using with Basic GEC."""
    def __init__(self, channels,GAMMA_full, wavetype='haar', std_pool_func=np.max):
        """Initialize BM3D_Basic_GEC

        Args:
            channels (int): number of channels to apply BM3D.
                If channels == 1, apply BM3D to the real part and scale the imaginary part by 0.1.
                If channels == 2, apply BM3D to both real and imaginary parts seperately.
            GAMMA_full (array): contains the GAMMA values in each subband of the wavelet
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.channels = channels
        self.std_pool_func = std_pool_func
        self.wavetype = wavetype
        self.GAMMA_full = GAMMA_full

    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)

    
        denoised = self._denoise(wavelet, variances)

        if calc_divergence:
            alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances):    
        level = wavelet.get_bands()
        std_pooled = self.std_pool_func(np.sqrt(variances))
        inp_dnz_foo = mutil.multiply_tau_subbandwise_full(wavelet, 1/np.sqrt(self.GAMMA_full))
        noisy_image = inp_dnz_foo.inverse()
        if self.channels == 1:
            noisy_real = noisy_image.real
            noisy_imag = noisy_image.imag
            denoised_real = torch.Tensor(bm3d(noisy_real, std_pooled))
            denoised_imag = noisy_imag * 0.1
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet_foo = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
            denoised_wavelet = mutil.multiply_tau_subbandwise_full(denoised_wavelet_foo, np.sqrt(self.GAMMA_full))
            
        elif self.channels == 2:
            noisy_real = noisy_image.real
            noisy_imag = noisy_image.imag
            denoised_real = torch.Tensor(bm3d(noisy_real, std_pooled))
            denoised_imag = torch.Tensor(bm3d(noisy_imag, std_pooled))
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet_foo = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
            denoised_wavelet = mutil.multiply_tau_subbandwise_full(denoised_wavelet_foo, np.sqrt(self.GAMMA_full))
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

    
    
class LMSE:
    """Wrapper of LMSE for using with Basic GEC."""
    def __init__(self, GAMMA_full, y, idx1_complement, idx2_complement,level,slices,sigma_w,N,wavetype='haar'):
        """Initialize LMSE

        Args:
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.wavetype = wavetype
        self.GAMMA_full = GAMMA_full
        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N

    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_2(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances):
        
        GAMMA_1_full = 1/variances
        A_Precon = mutil.preconditioned_A_op_fft_radial_full(self.idx1_complement,self.idx2_complement,self.GAMMA_full,self.wavetype,self.level,self.slices,GAMMA_1_full, self.sigma_w,self.N) 
    
        y_precon_0 = mutil.multiply_tau_subbandwise_full(wavelet, self.sigma_w*np.sqrt(GAMMA_1_full))
        y_precon_1 = y_precon_0.pyramid_forward(get_slices=False, to_tensor=False)
        y_precon = mutil.precon_3Dmat_to_vec(np.stack((self.y,y_precon_1)),self.N)

        x_1_bar_cap_t_vec, istop, itn, normr = scipy.sparse.linalg.lsqr(A_Precon, y_precon, atol=1e-6, btol=1e-6 ,iter_lim=200,show=False)[:4]
#         print('itn LMSE : ', itn)

        x_1_bar_cap_t_mat = np.reshape(x_1_bar_cap_t_vec,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        denoised_wavelet = mutil.multiply_tau_subbandwise_full(tutil.pyramid_backward(x_1_bar_cap_t_mat, self.slices), np.sqrt(self.GAMMA_full))

        return denoised_wavelet

class LMSE_Simplified_GEC:
    """Wrapper of LMSE for using with Simplified GEC."""
    def __init__(self, y, idx1_complement, idx2_complement,level,slices,sigma_w,N,LMSE_inner_iter_lim, beta_tune_LMSE = 1, wavetype='haar'):
        """Initialize LMSE

        Args:
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.wavetype = wavetype
        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        
    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        variances *= self.beta_tune_LMSE
#         print('beta tune LMSE : ', self.beta_tune_LMSE)
        
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_3_complex(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances):
        
        GAMMA_1_full = 1/variances
        A_Precon = mutil.preconditioned_A_op_fft_radial_Simplified_GEC_full(self.idx1_complement,self.idx2_complement,GAMMA_1_full,self.wavetype,self.level,self.slices, self.sigma_w,self.N) 
        y_precon_0 = mutil.multiply_tau_subbandwise_full(wavelet, self.sigma_w*np.sqrt(GAMMA_1_full))
        y_precon_1 = y_precon_0.pyramid_forward(get_slices=False, to_tensor=False)
        y_precon = mutil.precon_3Dmat_to_vec(np.stack((self.y,y_precon_1)),self.N)

#         print(y_precon.shape)
#         print((A_Precon._rmatvec(y_precon)).shape)
#         print((A_Precon._matvec(A_Precon._rmatvec(y_precon))).shape)       
        
        

        x_1_bar_cap_t_vec, istop, itn, normr = scipy.sparse.linalg.lsqr(A_Precon, y_precon, atol=1e-6, btol=1e-6 ,iter_lim=self.LMSE_inner_iter_lim,show=False)[:4]
#         print('itn LMSE : ', itn)

        x_1_bar_cap_t_mat = np.reshape(x_1_bar_cap_t_vec,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        denoised_wavelet = tutil.pyramid_backward(x_1_bar_cap_t_mat, self.slices)

        return denoised_wavelet

    
class LMSE_Simplified_GEC_pre_conditioned_div:
    # Trying to use Prof Schniter's poster type formula to compute divergence
    """Wrapper of LMSE for using with Simplified GEC."""
    def __init__(self, y, idx1_complement, idx2_complement,level,slices,sigma_w,N,LMSE_inner_iter_lim, beta_tune_LMSE = 1, wavetype='haar'):
        """Initialize LMSE

        Args:
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.wavetype = wavetype
        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        
    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        variances *= self.beta_tune_LMSE
#         print('beta tune LMSE : ', self.beta_tune_LMSE)
        
        denoised = self._denoise(wavelet, variances,self.y)
        if calc_divergence:
            alpha = calc_MC_LMSE_divergence_complex(self._denoise, denoised, wavelet, variances,self.y,self.wavetype)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances, y_denoiser):
        
        GAMMA_1_full = 1/variances
        A_Precon = mutil.preconditioned_A_op_fft_radial_Simplified_GEC_full(self.idx1_complement,self.idx2_complement,GAMMA_1_full,self.wavetype,self.level,self.slices, self.sigma_w,self.N) 
        y_precon_0 = mutil.multiply_tau_subbandwise_full(wavelet, self.sigma_w*np.sqrt(GAMMA_1_full))
        y_precon_1 = y_precon_0.pyramid_forward(get_slices=False, to_tensor=False)
        y_precon = mutil.precon_3Dmat_to_vec(np.stack((y_denoiser,y_precon_1)),self.N)

#         print(y_precon.shape)
#         print((A_Precon._rmatvec(y_precon)).shape)
#         print((A_Precon._matvec(A_Precon._rmatvec(y_precon))).shape)       
        
        

        x_1_bar_cap_t_vec, istop, itn, normr = scipy.sparse.linalg.lsqr(A_Precon, y_precon, atol=1e-6, btol=1e-6 ,iter_lim=self.LMSE_inner_iter_lim,show=False)[:4]
#         print('itn LMSE preconditioned : ', itn)
#         print(x_1_bar_cap_t_vec.shape)
        x_1_bar_cap_t_mat = np.reshape(x_1_bar_cap_t_vec,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        denoised_wavelet = tutil.pyramid_backward(x_1_bar_cap_t_mat, self.slices)

        return denoised_wavelet

    
    
    
class LMSE_multicoil_Simplified_GEC:
    """Wrapper of LMSE for using with Simplified GEC."""
    def __init__(self, y, idx1_complement, idx2_complement,sens_map, level,slices,sigma_w,N,nc, LMSE_inner_iter_lim, beta_tune_LMSE = 1, wavetype='haar'):
        """Initialize LMSE

        Args:
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.wavetype = wavetype
        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sens_map = sens_map
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
        self.nc = nc
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        
    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        variances *= self.beta_tune_LMSE
#         print('beta tune LMSE : ', self.beta_tune_LMSE)

        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_3_complex(self._denoise, denoised, wavelet, variances)
#             alpha = calc_MC_LMSE_divergence_complex(self._denoise, denoised, wavelet, variances,self.y,self.wavetype)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances):
        
        GAMMA_1_full = 1/variances
        A_Precon = mutil.preconditioned_multicoil_A_op_fft_Simplified_GEC_full(self.idx1_complement,self.idx2_complement,GAMMA_1_full,self.wavetype,self.level,self.slices, self.sigma_w,self.N,self.sens_map) 
        
        
        y_precon_0 = mutil.multiply_tau_subbandwise_full(wavelet, self.sigma_w*np.sqrt(GAMMA_1_full))
        y_precon_1 = y_precon_0.pyramid_forward(get_slices=False, to_tensor=False)
        
        y_precon = mutil.precon_3Dmat_to_vec_multi_coil(self.y,y_precon_1,self.N,self.nc)
        
#         print(y_precon.shape)
        
#         print((A_Precon._rmatvec(y_precon)).shape)
#         print((A_Precon._matvec(A_Precon._rmatvec(y_precon))).shape)
        
#         A_Precon_A = aslinearoperator(A_Precon)
#         print('A_Precon_A.shape: ',A_Precon_A.shape )
        
        x_1_bar_cap_t_vec, istop, itn, normr = scipy.sparse.linalg.lsqr(A_Precon, y_precon, atol=1e-6, btol=1e-6 ,iter_lim=self.LMSE_inner_iter_lim,show=False)[:4]
#         print('itn LMSE : ', itn)

        x_1_bar_cap_t_mat = np.reshape(x_1_bar_cap_t_vec,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        denoised_wavelet = tutil.pyramid_backward(x_1_bar_cap_t_mat, self.slices)

        return denoised_wavelet

class LMSE_multicoil_Simplified_GEC_2:
    """Wrapper of LMSE for using with Simplified GEC."""
    def __init__(self, y, idx1_complement, idx2_complement,sens_map, level,slices,sigma_w,N,nc, LMSE_inner_iter_lim, beta_tune_LMSE = 1, wavetype='haar'):
        """Initialize LMSE

        Args:
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.wavetype = wavetype
        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sens_map = sens_map
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
        self.nc = nc
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        
    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        variances *= self.beta_tune_LMSE
#         print('beta tune LMSE : ', self.beta_tune_LMSE)

        denoised = self._denoise(wavelet, variances,self.y)
        if calc_divergence:
#             alpha = calc_MC_divergence_2(self._denoise, denoised, wavelet, variances)
            alpha = calc_MC_LMSE_divergence_complex(self._denoise, denoised, wavelet, variances,self.y,self.wavetype)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances,y_denoiser):
        
        GAMMA_1_full = 1/variances
        A_Precon = mutil.preconditioned_multicoil_A_op_fft_Simplified_GEC_full(self.idx1_complement,self.idx2_complement,GAMMA_1_full,self.wavetype,self.level,self.slices, self.sigma_w,self.N,self.sens_map) 
        
        
        y_precon_0 = mutil.multiply_tau_subbandwise_full(wavelet, self.sigma_w*np.sqrt(GAMMA_1_full))
        y_precon_1 = y_precon_0.pyramid_forward(get_slices=False, to_tensor=False)
        
        y_precon = mutil.precon_3Dmat_to_vec_multi_coil(y_denoiser,y_precon_1,self.N,self.nc)
        
#         print(y_precon.shape)
        
#         print((A_Precon._rmatvec(y_precon)).shape)
#         print((A_Precon._matvec(A_Precon._rmatvec(y_precon))).shape)
        
#         A_Precon_A = aslinearoperator(A_Precon)
#         print('A_Precon_A.shape: ',A_Precon_A.shape )
        
        x_1_bar_cap_t_vec, istop, itn, normr = scipy.sparse.linalg.lsqr(A_Precon, y_precon, atol=1e-6, btol=1e-6 ,iter_lim=self.LMSE_inner_iter_lim,show=False)[:4]
#         print('itn LMSE : ', itn)

        x_1_bar_cap_t_mat = np.reshape(x_1_bar_cap_t_vec,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        denoised_wavelet = tutil.pyramid_backward(x_1_bar_cap_t_mat, self.slices)

        return denoised_wavelet

    
    
    
    
    
    
    
class LMSE_Simplified_GEC_real:
    """Wrapper of LMSE for using with Simplified GEC."""
    def __init__(self, y, idx1_complement, idx2_complement,level,slices,sigma_w,N,LMSE_inner_iter_lim,wavetype='haar'):
        """Initialize LMSE

        Args:
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.wavetype = wavetype
        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim

    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_2_real(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances):
        
        GAMMA_1_full = 1/variances
        A_Precon = mutil.preconditioned_A_op_fft_radial_Simplified_GEC_full(self.idx1_complement,self.idx2_complement,GAMMA_1_full,self.wavetype,self.level,self.slices, self.sigma_w,self.N) 
        y_precon_0 = mutil.multiply_tau_subbandwise_full(wavelet, self.sigma_w*np.sqrt(GAMMA_1_full))
        y_precon_1 = y_precon_0.pyramid_forward(get_slices=False, to_tensor=False)
        y_precon = mutil.precon_3Dmat_to_vec(np.stack((self.y,y_precon_1)),self.N)

        x_1_bar_cap_t_vec, istop, itn, normr = scipy.sparse.linalg.lsqr(A_Precon, y_precon, atol=1e-6, btol=1e-6 ,iter_lim=self.LMSE_inner_iter_lim,show=False)[:4]
#         print('itn LMSE : ', itn)

        x_1_bar_cap_t_mat = np.reshape(np.real(x_1_bar_cap_t_vec),(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        denoised_wavelet = tutil.pyramid_backward(x_1_bar_cap_t_mat, self.slices)

        return denoised_wavelet

    
    
    
class ColoredDnCNN_GEC:
    """Wrapper of algorithm.denoiser.ColoredDnCNN for using with Basic GEC.
    
    Note:
        In D-VDAMP, the noisy wavelets have complex coefficients.

        For real ground truths (model channel = 1), we appply the denoiser to
        only the real part and scale the imaginary part by 0.1.

        For complex ground truths (model channel = 2), we pass a tensor where
        the first channel is the real part and the second channel is the imaginary part
        to the model. However, this feature is not currently supported.
    """
    def __init__(self, modeldir, std_ranges, GAMMA_full, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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
        self.models = self._load_models(modeldir, num_layers, std_channels)
        self.std_pool_func = std_pool_func
        self.verbose = verbose
        self.GAMMA_full = GAMMA_full

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
        variances *= gamma
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = torch.from_numpy(variances).sqrt().to(device=self.device, dtype=torch.float32)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
        
        
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('ColoredDnCNN_VDAMP select: {}'.format(select))

        # Denoise
        
        inp_dnz_foo = mutil.multiply_tau_subbandwise_full(wavelet, 1/np.sqrt(self.GAMMA_full))
        noisy_image = inp_dnz_foo.inverse().unsqueeze(0)
        
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            denoised_real = self.models[select](noisy_real, stds).cpu()
            denoised_imag = noisy_imag * 0.1
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet_foo = tutil.forward(denoised_image[0], wavelet=self.wavetype, level=level)
            denoised_wavelet = mutil.multiply_tau_subbandwise_full(denoised_wavelet_foo, np.sqrt(self.GAMMA_full))
            
        elif self.channels == 2:
            noisy_image = torch.vstack([noisy_image.real, noisy_image.imag])
            noisy_image = noisy_image.to(device=self.device, dtype=torch.float32)
            denoised_image = self.models[select](noisy_image, stds).cpu()
            denoised_image = denoised_image[0] + 1j * denoised_image[1]
            denoised_wavelet_foo = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
            denoised_wavelet = mutil.multiply_tau_subbandwise_full(denoised_wavelet_foo, np.sqrt(self.GAMMA_full))
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

    def _load_models(self, modeldirs, num_layers, std_channels):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = ColoredDnCNN(channels=self.channels, num_layers=num_layers, std_channels=std_channels)
            load_checkpoint(modeldir, model, None, device=self.device)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    
def correct_GAMMA_using_EM(denoiser,noisy_wavelet, variances, num_em_iter):

    """Correct the variances entering the denoiser using EM iterations

    """
    level = noisy_wavelet.get_bands()
    
#     print('running EM correction')
    
    for i in range(num_em_iter):
        
#         variances_subband = tutil.reformat_array2subband(variances)
        
        denoised, alpha = denoiser(noisy_wavelet, variances)
        
#         variance_new_subband = [None] * (level + 1)

#         variance_new_subband[0] = (np.sum(np.abs(denoised.coeff[0] - noisy_wavelet.coeff[0])**2)/ noisy_wavelet.coeff[0].size) + (alpha[0]*variances_subband[0])
        
        

#         for s in range(level):
#             variance_new_subband[s + 1] = 3 * [None]
#             for b in range(3):
#                 variance_new_subband[s + 1][b] = (np.sum(np.abs(denoised.coeff[s + 1][b] - noisy_wavelet.coeff[s + 1][b])**2)/ noisy_wavelet.coeff[s + 1][b].size) + (alpha[s + 1][b]*variances_subband[s + 1][b])

#         variances = tutil.reformat_subband2array(variance_new_subband)
        
        variances = tutil.reformat_subband2array(_calc_mse(denoised,noisy_wavelet)) + tutil.reformat_subband2array(alpha)*variances
        
    return variances



def calc_MC_LMSE_divergence_complex_with_warm_start(denoiser, denoised, wavelet, variances, warm_start_div_mat, y_dummy, wavetype,N):
    # Added Warm Start provedure in this to speed things up
#     print('complex divergence computation +-1 for LMSE!')
    # modified by Saurav

    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    
    probing_vec = np.zeros(y_dummy.shape,dtype=np.complex128) 

    wavelet_jittered = tutil.forward(probing_vec, wavelet=wavetype, level=level)

#     wavelet_jittered = wavelet.copy()
    
    noise_vec = np.sqrt(1/2)*(np.sign(np.random.randn(*wavelet_jittered.coeff[0].shape)) + 1j * np.sign(np.random.randn(*wavelet_jittered.coeff[0].shape)))
    
    wavelet_jittered.coeff[0] += noise_vec
    
    zero_y = y_dummy*0
    
    div_mat = np.zeros((N,3*level+1),dtype=np.complex128)
    
    denoised_jittered, div_mat[:,0] = denoiser(wavelet_jittered, variances, np.squeeze(warm_start_div_mat[:,0]), zero_y)
    
    alpha[0] = 1. / wavelet_jittered.coeff[0].size * np.real(np.dot( np.conj(noise_vec).reshape(-1),(denoised_jittered.coeff[0]).reshape(-1)))
    
    count = 1
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            
            probing_vec = np.zeros(y_dummy.shape,dtype=np.complex128)
            wavelet_jittered = tutil.forward(probing_vec, wavelet=wavetype, level=level)

            noise_vec = np.sqrt(1/2)*(np.sign(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)) + 1j * np.sign(np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)))
            
            
            wavelet_jittered.coeff[s + 1][b] += noise_vec

            denoised_jittered, div_mat[:,count] = denoiser(wavelet_jittered, variances, np.squeeze(warm_start_div_mat[:,count]),zero_y)
            count = count + 1
            
            alpha[s + 1][b] = 1. / wavelet_jittered.coeff[s + 1][b].size * np.real(np.dot(np.conj(noise_vec).reshape(-1), (denoised_jittered.coeff[s + 1][b]).reshape(-1))) 
                    
    return alpha, div_mat



class LMSE_Simplified_GEC_pre_conditioned_div_with_warm_start:
    # Trying to use Prof Schniter's poster type formula to compute divergence
    # Added Warm Start provedure in this to speed things up
    """Wrapper of LMSE for using with Simplified GEC."""
    def __init__(self, y, idx1_complement, idx2_complement,level,slices,sigma_w,N,LMSE_inner_iter_lim, beta_tune_LMSE = 1, wavetype='haar'):
        """Initialize LMSE

        Args:
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.wavetype = wavetype
        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.level = level
        self.slices = slices
        self.sigma_w = sigma_w
        self.N = N
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        
    def __call__(self, wavelet, variances, warm_start_vector, warm_start_div_mat, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        variances *= self.beta_tune_LMSE
#         print('beta tune LMSE : ', self.beta_tune_LMSE)
        
        denoised, x_1_bar_cap_t_vec = self._denoise(wavelet, variances, np.squeeze(warm_start_vector),self.y)
        
        if calc_divergence:
            alpha, div_mat = calc_MC_LMSE_divergence_complex_with_warm_start(self._denoise, denoised, wavelet, variances,warm_start_div_mat,self.y,self.wavetype,self.N)
            return denoised, alpha, x_1_bar_cap_t_vec, div_mat
        else:
            return denoised, x_1_bar_cap_t_vec, div_mat

    def _denoise(self, wavelet, variances, warm_start_vector, y_denoiser):
        
        GAMMA_1_full = 1/variances
        A_Precon = mutil.preconditioned_A_op_fft_radial_Simplified_GEC_full(self.idx1_complement,self.idx2_complement,GAMMA_1_full,self.wavetype,self.level,self.slices, self.sigma_w,self.N) 
        y_precon_0 = mutil.multiply_tau_subbandwise_full(wavelet, self.sigma_w*np.sqrt(GAMMA_1_full))
        y_precon_1 = y_precon_0.pyramid_forward(get_slices=False, to_tensor=False)
        y_precon = mutil.precon_3Dmat_to_vec(np.stack((y_denoiser,y_precon_1)),self.N)

#         print(y_precon.shape)
#         print((A_Precon._rmatvec(y_precon)).shape)
#         print((A_Precon._matvec(A_Precon._rmatvec(y_precon))).shape)       
        
        
#         print(warm_start_vector.shape)
        x_1_bar_cap_t_vec, istop, itn, normr = scipy.sparse.linalg.lsqr(A_Precon, y_precon, atol=1e-6, btol=1e-6 ,iter_lim=self.LMSE_inner_iter_lim,show=False,x0 = warm_start_vector)[:4]
#         print('itn LMSE preconditioned : ', itn)

        x_1_bar_cap_t_mat = np.reshape(x_1_bar_cap_t_vec,(int(np.sqrt(self.N)),int(np.sqrt(self.N))))
        
        denoised_wavelet = tutil.pyramid_backward(x_1_bar_cap_t_mat, self.slices)

        return denoised_wavelet, x_1_bar_cap_t_vec

    
    
class DnCNN_cpc_VDAMP_real:
    """
    
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = 1, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False, level = 4):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence_2_real(self._denoise, denoised, wavelet, variances)
#             alpha = calc_MC_divergence_3_complex(self._denoise, denoised, wavelet, variances)
#             print('Note: using multiple iterations for computing divergence!')
#             alpha = calc_MC_divergence_multiple_iteration_real(self._denoise, denoised, wavelet, variances)


            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = np.sqrt(variances)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('DnCNN_cpc_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = (wavelet.inverse().unsqueeze(0)).unsqueeze(0)
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            
            noisy_image_dncnn_cpc = concatenate_noisy_data_with_a_noise_realization_of_given_stds(noisy_real.cpu(),stds, self.wavetype, self.level)
            
            noisy_image_dncnn_cpc = noisy_image_dncnn_cpc.to(device=self.device, dtype=torch.float32)
            
            denoised_real = self.models[select](noisy_image_dncnn_cpc).cpu()
            
            denoised_image = denoised_real
            denoised_wavelet = tutil.forward(denoised_image[0,0], wavelet=self.wavetype, level=level)
            
        else:
            raise ValueError('Only support channel == 1')
        return denoised_wavelet

    def _load_models(self, modeldirs):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = load_model(modeldir)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    
    
class DnCNN_cpc_VDAMP_complex:
    """
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = 1, complex_weight = 0.1, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False, level = 4):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            
            alpha = calc_MC_divergence_3_complex(self._denoise, denoised, wavelet, variances)
            
            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = np.sqrt(variances)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('DnCNN_cpc_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = (wavelet.inverse().unsqueeze(0)).unsqueeze(0)
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            
            noisy_image_dncnn_cpc = concatenate_noisy_data_with_a_noise_realization_of_given_stds(noisy_real.cpu(),stds, self.wavetype, self.level)
            
            noisy_image_dncnn_cpc = noisy_image_dncnn_cpc.to(device=self.device, dtype=torch.float32)
            
            denoised_real = self.models[select](noisy_image_dncnn_cpc).cpu()
            
            denoised_imag = noisy_imag * self.complex_weight
            
            denoised_image = denoised_real + 1j * denoised_imag
            
#             print(denoised_image.shape)
#             print('normalizing')
#             denoised_image[0] = denoised_image[0]/torch.max(torch.abs(denoised_image[0]))

            denoised_wavelet = tutil.forward(denoised_image[0,0], wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1')
        return denoised_wavelet

    def _load_models(self, modeldirs):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = load_model(modeldir)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

    
class DnCNN_cpc_VDAMP:
    """Wrapper of algorithm.denoiser.ColoredDnCNN for using with D-VDAMP.
    
    Note:
        In D-VDAMP, the noisy wavelets have complex coefficients.

        For real ground truths (model channel = 1), we appply the denoiser to
        only the real part and scale the imaginary part by 0.1.

        For complex ground truths (model channel = 2), we pass a tensor where
        the first channel is the real part and the second channel is the imaginary part
        to the model. However, this feature is not currently supported.
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, beta_tune = 1, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False, level = 4):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
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
        self.beta_tune = beta_tune
        self.verbose = verbose
        self.level = level

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
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
#         variances *= gamma
        variances *= self.beta_tune
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = np.sqrt(variances)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
#         select = select - 1 # Saurav added for testing
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('DnCNN_cpc_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = (wavelet.inverse().unsqueeze(0)).unsqueeze(0)
        if self.channels == 1:
            
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            
            noisy_image_dncnn_cpc = concatenate_noisy_data_with_a_noise_realization_of_given_stds(noisy_real.cpu(),stds, self.wavetype, self.level)
            
            noisy_image_dncnn_cpc = noisy_image_dncnn_cpc.to(device=self.device, dtype=torch.float32)
            
            denoised_real = self.models[select](noisy_image_dncnn_cpc).cpu()
            
            denoised_imag = noisy_imag * 0.1
            
            denoised_image = denoised_real + 1j * denoised_imag
            
#             print(denoised_image.shape)
#             print('normalizing')
#             denoised_image[0] = denoised_image[0]/torch.max(torch.abs(denoised_image[0]))

            denoised_wavelet = tutil.forward(denoised_image[0,0], wavelet=self.wavetype, level=level)

        else:
            raise ValueError('Only support channel == 1.')
        return denoised_wavelet

    def _load_models(self, modeldirs):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = load_model(modeldir)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models
