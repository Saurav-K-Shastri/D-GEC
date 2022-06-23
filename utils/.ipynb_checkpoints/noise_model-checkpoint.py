import torch
import numpy as np
from utils import transform as tutil

def get_noise(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    noise = torch.randn_like(data);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
    else:
        noise.data = noise.data * noise_std;
    return noise

def get_noisy_data_with_SD_map(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    

    noise = torch.randn_like(data);
    
    result_data = torch.zeros(data.shape[0],2,data.shape[2],data.shape[3]);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
            result_data[i,0,:,:] = data[i,0,:,:] + noise.data[i]
            result_data[i,1,:,:] = noise_tensor_array[i]*torch.ones(data[i,0,:,:].shape)
    else:
        noise.data = noise.data * noise_std;
        n = noise.shape[0];
        for i in range(n):
            result_data[i,0,:,:] = data[i,0,:,:] + noise.data[i]
            result_data[i,1,:,:] = noise_std*torch.ones(data[i,0,:,:].shape)
        
    return result_data

def get_noisy_data_with_SD_map_complex(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    

    noise = torch.randn_like(data);
    
    result_data = torch.zeros(data.shape[0],3,data.shape[2],data.shape[3]);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (1/torch.sqrt(torch.tensor(2)))*((max_noise - min_noise) * torch.rand(n) + min_noise);
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
            result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
            result_data[i,2,:,:] = noise_tensor_array[i]*torch.ones(data[i,0,:,:].shape)
    else:
        noise.data = (1/torch.sqrt(torch.tensor(2))) * noise.data * noise_std;
        n = noise.shape[0];
        for i in range(n):
            result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
            result_data[i,2,:,:] = (1/torch.sqrt(torch.tensor(2)))*noise_std*torch.ones(data[i,0,:,:].shape)
        
    return result_data


def get_noisy_data_with_noise_map(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    

    noise = torch.randn_like(data);
    noise_2 = torch.randn_like(data);
    
    result_data = torch.zeros(data.shape[0],2,data.shape[2],data.shape[3]);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
            noise_2.data[i] = noise_2.data[i] * noise_tensor_array[i];
            result_data[i,0,:,:] = data[i,0,:,:] + noise.data[i]
            result_data[i,1,:,:] = noise_2.data[i]
    else:
        noise.data = noise.data * noise_std;
        noise_2.data = noise_2.data * noise_std;
        n = noise.shape[0];
        for i in range(n):
            result_data[i,0,:,:] = data[i,0,:,:] + noise.data[i]
            result_data[i,1,:,:] = noise_2.data[i]
        
    return result_data

def get_noisy_data_with_noise_map_complex(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    

    noise = torch.randn_like(data);
    noise_2 = torch.randn(data.shape[0],1,data.shape[2],data.shape[3]);
    
    result_data = torch.zeros(data.shape[0],3,data.shape[2],data.shape[3]);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (1/torch.sqrt(torch.tensor(2)))*((max_noise - min_noise) * torch.rand(n) + min_noise);
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
            noise_2.data[i] = noise_2.data[i] * noise_tensor_array[i];
            result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
            result_data[i,2,:,:] = noise_2.data[i]
    else:
        noise.data = (1/torch.sqrt(torch.tensor(2))) * noise.data * noise_std;
        noise_2.data = (1/torch.sqrt(torch.tensor(2))) * noise_2.data * noise_std;
        n = noise.shape[0];
        for i in range(n):
            result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
            result_data[i,2,:,:] = noise_2.data[i]
        
    return result_data

    

def get_noise_complex(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    noise = torch.randn_like(data);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (1/torch.sqrt(torch.tensor(2)))*((max_noise - min_noise) * torch.rand(n) + min_noise);
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
    else:
        noise.data = (1/torch.sqrt(torch.tensor(2))) * noise.data * noise_std;
    return noise

def get_noise_for_combined_data(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    noise = torch.randn_like(data);
    if mode == 'B':
        n = noise.shape[0]*noise.shape[1]
        noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
        count = 0
#         print('n :', n)
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                noise.data[i,j] = noise.data[i,j] * noise_tensor_array[count];
                count = count + 1
    else:
        noise.data = noise.data * noise_std;
#     print('count :', count)
    return noise


def get_noisy_data_and_noise_with_same_stat(data, min_noise , max_noise, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],2,data.shape[2],data.shape[3]);
    
    for i in range(data.shape[0]):
        [result_data[i,0,:,:], result_data[i,1,:,:]] = generate_noisy_image_and_noise(data[i,0,:,:].numpy(),wavetype,level, min_noise, max_noise)
    
    return result_data

def get_noisy_data_and_stds(data, min_noise , max_noise, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],1,data.shape[2],data.shape[3]);
    stds = torch.zeros(data.shape[0],3*level + 1)
    
    for i in range(data.shape[0]):
        result_data[i,0,:,:],stds[i,:] = generate_noisy_image_and_noise_stds(data[i,0,:,:].numpy(),wavetype,level, min_noise, max_noise)
    
    return result_data, stds



def get_noisy_data_and_noise_with_same_stat_log_uniform(data, min_noise , max_noise, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],2,data.shape[2],data.shape[3]);
    
    for i in range(data.shape[0]):
        [result_data[i,0,:,:], result_data[i,1,:,:]] = generate_noisy_image_and_noise_log_uniform(data[i,0,:,:].numpy(),wavetype,level, min_noise, max_noise)
    
    return result_data


def get_noisy_data_and_five_noise_with_same_stat(data, min_noise , max_noise, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],6,data.shape[2],data.shape[3]);
    
    for i in range(data.shape[0]):
        [result_data[i,0,:,:], result_data[i,1,:,:], result_data[i,2,:,:], result_data[i,3,:,:], result_data[i,4,:,:], result_data[i,5,:,:]] = generate_noisy_image_and_five_noise(data[i,0,:,:].numpy(),wavetype,level, min_noise, max_noise)
    
    return result_data

def get_noisy_data_and_five_noise_with_same_stat_known_stds(data, stds, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],6,data.shape[2],data.shape[3]);
    
    for i in range(data.shape[0]):
        [result_data[i,0,:,:], result_data[i,1,:,:], result_data[i,2,:,:], result_data[i,3,:,:], result_data[i,4,:,:], result_data[i,5,:,:]] = generate_noisy_image_and_five_noise_known_stds(data[i,0,:,:].numpy(),wavetype,level, stds)
    
    return result_data


def get_noisy_data_correlated_noise(data, min_noise , max_noise, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],1,data.shape[2],data.shape[3]);
    dummy = torch.zeros(data.shape[0],1,data.shape[2],data.shape[3]);
    
    for i in range(data.shape[0]):
        [result_data[i,0,:,:], dummy] = generate_noisy_image_and_noise(data[i,0,:,:].numpy(),wavetype,level, min_noise, max_noise)
    
    return result_data


def get_noisy_data_and_noise_with_same_stat_known_stds(data, stds, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],2,data.shape[2],data.shape[3]);
    
    for i in range(data.shape[0]):
        [result_data[i,0,:,:], result_data[i,1,:,:]] = generate_noisy_image_and_noise_known_stds(data[i,0,:,:].numpy(),wavetype,level, stds)
    
    return result_data

def generate_noisy_image_and_noise(image,wavetype,level, min_noise, max_noise):
    wavelet = tutil.forward(image, wavelet=wavetype, level=level)
    num_stds = 3*level + 1
    stds = torch.FloatTensor(num_stds).uniform_(min_noise, max_noise)
    
    noisy_wavelet, noise = tutil.get_noise_subbandwise_for_dncnn_cn(wavelet, stds, is_complex=False)
    
    noisy_image = noisy_wavelet.inverse(wavelet=wavetype)
    noise_in_image = noise.inverse(wavelet=wavetype)
    return noisy_image, noise_in_image


def generate_noisy_image_and_noise_stds(image,wavetype,level, min_noise, max_noise):
    wavelet = tutil.forward(image, wavelet=wavetype, level=level)
    num_stds = 3*level + 1
    stds = torch.FloatTensor(num_stds).uniform_(min_noise, max_noise)
    
    noisy_wavelet, noise = tutil.get_noise_subbandwise_for_dncnn_cn(wavelet, stds, is_complex=False)
    
    noisy_image = noisy_wavelet.inverse(wavelet=wavetype)

    return noisy_image, stds




def generate_noisy_image_and_noise_log_uniform(image,wavetype,level, min_noise, max_noise):
    wavelet = tutil.forward(image, wavelet=wavetype, level=level)
    num_stds = 3*level + 1
#     stds = torch.FloatTensor(num_stds).uniform_(min_noise, max_noise)
    stds = torch.exp(torch.FloatTensor(num_stds).uniform_(np.log(min_noise),np.log(max_noise)))
    
    noisy_wavelet, noise = tutil.get_noise_subbandwise_for_dncnn_cn(wavelet, stds, is_complex=False)
    
    noisy_image = noisy_wavelet.inverse(wavelet=wavetype)
    noise_in_image = noise.inverse(wavelet=wavetype)
    return noisy_image, noise_in_image





def generate_noisy_image_and_five_noise(image,wavetype,level, min_noise, max_noise):
    wavelet = tutil.forward(image, wavelet=wavetype, level=level)
    num_stds = 3*level + 1
    stds = torch.FloatTensor(num_stds).uniform_(min_noise, max_noise)
    
    noisy_wavelet, noise1, noise2, noise3, noise4, noise5 = tutil.get_noise_subbandwise_for_dncnn_cn_five(wavelet, stds, is_complex=False)
    
    noisy_image = noisy_wavelet.inverse(wavelet=wavetype)
    noise_in_image1 = noise1.inverse(wavelet=wavetype)
    noise_in_image2 = noise2.inverse(wavelet=wavetype)
    noise_in_image3 = noise3.inverse(wavelet=wavetype)
    noise_in_image4 = noise4.inverse(wavelet=wavetype)
    noise_in_image5 = noise5.inverse(wavelet=wavetype)
    
    return noisy_image, noise_in_image1, noise_in_image2, noise_in_image3, noise_in_image4, noise_in_image5



def generate_noisy_image_and_noise_known_stds(image,wavetype,level, stds_inp):
    wavelet = tutil.forward(image, wavelet=wavetype, level=level)
    num_stds = 3*level + 1
    stds = torch.FloatTensor(stds_inp)
    
    noisy_wavelet, noise = tutil.get_noise_subbandwise_for_dncnn_cn(wavelet, stds, is_complex=False)
    
    noisy_image = noisy_wavelet.inverse(wavelet=wavetype)
    noise_in_image = noise.inverse(wavelet=wavetype)
    return noisy_image, noise_in_image


def generate_noisy_image_and_five_noise_known_stds(image,wavetype,level, stds_inp):
    wavelet = tutil.forward(image, wavelet=wavetype, level=level)
    num_stds = 3*level + 1
    stds = torch.FloatTensor(stds_inp)
    
    noisy_wavelet, noise1, noise2, noise3, noise4, noise5 = tutil.get_noise_subbandwise_for_dncnn_cn_five(wavelet, stds, is_complex=False)
    
    noisy_image = noisy_wavelet.inverse(wavelet=wavetype)
    noise_in_image1 = noise1.inverse(wavelet=wavetype)
    noise_in_image2 = noise2.inverse(wavelet=wavetype)
    noise_in_image3 = noise3.inverse(wavelet=wavetype)
    noise_in_image4 = noise4.inverse(wavelet=wavetype)
    noise_in_image5 = noise5.inverse(wavelet=wavetype)
    
    return noisy_image, noise_in_image1, noise_in_image2, noise_in_image3, noise_in_image4, noise_in_image5


def get_true_noise_realization_for_blind_cpc(noise,wavetype,level, feta):
    
    wavelet = tutil.forward(noise, wavelet=wavetype, level=level)
    SD_computed = np.sqrt(tutil.reformat_subband2array(calc_var(wavelet)))*feta
    
    noisy_image, noise_in_image = generate_noisy_image_and_noise_known_stds(noise,wavetype,level, SD_computed)
    
    
    return noise_in_image, SD_computed
    
    

def calc_var(test):
    """Calculate variance of noise.

    """
    mse = [None] * (test.get_bands() + 1)
    mse[0] = np.mean(np.abs(test.coeff[0]) ** 2)
    for b in range(1, test.get_bands() + 1):
        mse_band = [None] * 3
        for s in range(3):
            mse_band[s] = np.mean(np.abs(test.coeff[b][s]) ** 2)
        mse[b] = mse_band
    return mse




def concatenate_noisy_data_with_a_noise_realization_of_given_stds(data, stds, wavetype, level):
    
    result_data = torch.zeros(data.shape[0],2,data.shape[2],data.shape[3]);
    
    [dummy, result_data[0,1,:,:]] = generate_noisy_image_and_noise_known_stds(data[0,0,:,:].numpy(),wavetype,level, stds)
    
    result_data[0,0,:,:] = data[0,0,:,:]
    
    return result_data