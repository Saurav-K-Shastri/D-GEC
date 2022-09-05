import torch
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
import time
from fastMRI_utils import transforms_new

def wave_forward_list(image, xfm):
    """
    return wavelet in generic [Yl,Yh] list
    image size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    
    """
#     device = image.device
#     xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    Yl, Yh = xfm(image)
    return [Yl, Yh]


def wave_inverse_list(wave_list, ifm):
    """
    return image of size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    
    """
    Yl, Yh = wave_list
#     device = Yl.device
#     ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)
    my_image = ifm((Yl, Yh))
    

    return my_image


def wave_forward_mat(image, xfm):
    """
    retruns wavelet matrix of same size as image
    image size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    
    """
#     device = image.device
#     xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    
#     t = time.time()
    Yl, Yh = xfm(image)
    


    my_wave_mat = get_wave_mat([Yl,Yh])

#     elapsed = time.time() - t
    
#     print('time take: ')
#     print(elapsed)
#     print(my_wave_mat.shape)

    return my_wave_mat


def wave_inverse_mat(wave_mat, ifm, level):
    """
    retruns image of size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    """
    Yl, Yh = wave_mat2list(wave_mat,level)
#     device = Yl.device
#     ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)
    my_image = ifm((Yl, Yh))
    return my_image



def get_wave_mat(wave_list):
    Yl,Yh = wave_list
    level = len(Yh)
    my_mat = Yl.clone()

    for i in range(level):
        index = level - 1 - i
        my_mat = torch.cat((torch.cat((my_mat,Yh[index][:,:,0,:,:].clone()),dim = 2),torch.cat((Yh[index][:,:,1,:,:].clone(),Yh[index][:,:,2,:,:].clone()),dim = 2)),dim = 3)
    return my_mat


def wave_mat2list(wave_mat,level = 4):
    
    assert wave_mat.shape[2] == wave_mat.shape[3], "last two dimentions should be same"
    
    N = wave_mat.shape[2]
    srt_LH_row = int(N/2)
    end_LH_row = N
    srt_LH_col = 0
    end_LH_col = int(N/2)
    
    Yh = []
    for i in range(level):
        Yh.append(torch.stack((wave_mat[:,:,srt_LH_row:end_LH_row,srt_LH_col:end_LH_col].clone(),wave_mat[:,:,srt_LH_col:end_LH_col,srt_LH_row:end_LH_row].clone(), wave_mat[:,:,srt_LH_row:end_LH_row,srt_LH_row:end_LH_row].clone()),dim = 2))
        end_LH_row = srt_LH_row
        srt_LH_row = int(srt_LH_row/2)
        srt_LH_col = 0
        end_LH_col = int(end_LH_col/2)

    Yl = wave_mat[:,:,0:end_LH_row,0:end_LH_row].clone()

    return [Yl,Yh]

def wave_add_list(wave_list_1,wave_list_2):
    
    Yl,Yh = wave_list_1
    level = len(Yh)
    
    wave_mat_1 = get_wave_mat(wave_list_1)
    wave_mat_2 = get_wave_mat(wave_list_2)
    
    sum_wave_mat = wave_mat_1 + wave_mat_2
    
    return wave_mat2list(sum_wave_mat,level)
    

def wave_sub_list(wave_list_1,wave_list_2):
    
    Yl,Yh = wave_list_1
    level = len(Yh)
    
    wave_mat_1 = get_wave_mat(wave_list_1)
    wave_mat_2 = get_wave_mat(wave_list_2)
    
    sub_wave_mat = wave_mat_1 - wave_mat_2
    
    return wave_mat2list(sub_wave_mat,level)

def wave_mul_list(wave_list_1,wave_list_2):
    """ element-wise multiplication"""
    Yl,Yh = wave_list_1
    level = len(Yh)
    
    wave_mat_1 = get_wave_mat(wave_list_1)
    wave_mat_2 = get_wave_mat(wave_list_2)
    
    mul_wave_mat = torch.mul(wave_mat_1, wave_mat_2)
    
    return wave_mat2list(mul_wave_mat,level)


def wave_aHb_dot_product_list(wave_list_a,wave_list_b):
    """ dot product of two wavelets"""
    Yl,Yh = wave_list_a
    level = len(Yh)
    
    wave_mat_a = get_wave_mat(wave_list_a)
    wave_mat_b = get_wave_mat(wave_list_b)
    
    dot_wave = torch.dot(torch.conj(torch.view_as_complex(wave_mat_a.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(wave_mat_b.permute(0,2,3,1).contiguous()).reshape(-1))
    
    return dot_wave



def wave_sum_list(wave_list):
    """ sum of all elements"""
    wave_mat = get_wave_mat(wave_list)
    result = torch.sum(torch.view_as_complex(wave_mat.permute(0,2,3,1).contiguous()))
    
    return result

def wave_scalar_mul_list(wave_list,scalar):
    """ scalar multiplication """
    Yl,Yh = wave_list
    level = len(Yh)
    
    wave_mat = get_wave_mat(wave_list)
    sclar_mul_mat = scalar*wave_mat
    
    return wave_mat2list(sclar_mul_mat,level)

def wave_scalar_mul_real_and_imag_list(wave_list,scalar_real,scalar_imag):
    """ scalar multiplication real and imag """
    Yl,Yh = wave_list
    level = len(Yh)
    
    wave_mat = get_wave_mat(wave_list)
    sclar_mul_mat[:,0,:,:] = scalar_real*wave_mat[:,0,:,:]
    sclar_mul_mat[:,1,:,:] = scalar_imag*wave_mat[:,1,:,:]
    
    return wave_mat2list(sclar_mul_mat,level)


def wave_scalar_mul_subbandwise_list(wave_list,scalar_vec):
    """ scalar vector multiplication i.e. multiplying each subband with the scalar correspinding to that subband """
    
    
    Yl,Yh = wave_list
    level = len(Yh)
    
    Yl_new,Yh_new = wave_mat2list(get_wave_mat(wave_list).clone())

    
    Yl_new = (Yl_new.permute(1,2,3,0)*scalar_vec[:,0]).permute(3,0,1,2)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            Yh_new[index][:,:,j,:,:] = (Yh_new[index][:,:,j,:,:].permute(1,2,3,0)*scalar_vec[:,count]).permute(3,0,1,2)
            count = count+1
    
    return [Yl_new,Yh_new]


def wave_scalar_mul_subbandwise_real_and_imag_list(wave_list,scalar_vec_real, scalar_vec_imag):
    """ scalar vector multiplication i.e. multiplying each subband with the scalar correspinding to that subband """
    
    
    Yl,Yh = wave_list
    level = len(Yh)
    
    Yl_new,Yh_new = wave_mat2list(get_wave_mat(wave_list).clone())

    
    Yl_new[:,0,:,:] = Yl_new[:,0,:,:]*scalar_vec_real[0]
    Yl_new[:,1,:,:] = Yl_new[:,1,:,:]*scalar_vec_imag[0]
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            Yh_new[index][:,0,j,:,:] = Yh_new[index][:,0,j,:,:]*scalar_vec_real[count]
            Yh_new[index][:,1,j,:,:] = Yh_new[index][:,1,j,:,:]*scalar_vec_imag[count]
            count = count+1
    
    return [Yl_new,Yh_new]


def add_noise_subbandwise_list_with_wave_mask(wave_list,stds,wave_mask_list):
    
    """ add noise of different noise levels to each subband with wave mask """
    Yl_mask,Yh_mask = wave_mask_list
    
    Yl,Yh = wave_list
    device = Yl.device
    level = len(Yh)
    
    is_complex = (Yh[0].shape[1] == 2)
    
    Yl_new,Yh_new = wave_mat2list(get_wave_mat(wave_list).clone())
    
    Yl_new = Yl_new + Yl_mask*generate_noise_mat(Yl.shape,stds[0,0],is_complex,device)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            Yh_new[index][:,:,j,:,:] = Yh_new[index][:,:,j,:,:] + Yh_mask[index][:,:,j,:,:]*generate_noise_mat(Yh_new[index][:,:,j,:,:].shape,stds[0,count],is_complex,device) 
            count = count+1
    
    return [Yl_new,Yh_new]



def add_noise_subbandwise_list(wave_list,stds):
    
    """ add noise of different noise levels to each subband """
    Yl,Yh = wave_list
    device = Yl.device
    level = len(Yh)
    
    is_complex = (Yh[0].shape[1] == 2)
    
    Yl_new,Yh_new = wave_mat2list(get_wave_mat(wave_list).clone())
    
    Yl_new = Yl_new + generate_noise_mat(Yl.shape,stds[0,0],is_complex,device)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            Yh_new[index][:,:,j,:,:] = Yh_new[index][:,:,j,:,:] + generate_noise_mat(Yh_new[index][:,:,j,:,:].shape,stds[0,count],is_complex,device) 
            count = count+1
    
    return [Yl_new,Yh_new]


def add_noise_subbandwise_list_batch(wave_list,stds):
    
    """ add noise of different noise levels to each subband """
    Yl,Yh = wave_list
    device = Yl.device
    level = len(Yh)
    
    is_complex = (Yh[0].shape[1] == 2)
    
    Yl_new,Yh_new = wave_mat2list(get_wave_mat(wave_list).clone())
    
    Yl_new = Yl_new + generate_noise_mat_batch(Yl.shape,stds[:,0],is_complex,device)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            Yh_new[index][:,:,j,:,:] = Yh_new[index][:,:,j,:,:] + generate_noise_mat_batch(Yh_new[index][:,:,j,:,:].shape,stds[:,count],is_complex,device) 
            count = count+1
    
    return [Yl_new,Yh_new]


def generate_noise_mat_batch(my_shape,my_std,is_complex,device):
    
    if is_complex:
        my_std_new = my_std.clone()/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = my_std.clone()
    
    my_noise_mat = (my_std_new*((torch.randn(my_shape, device = device)).permute(1,2,3,0))).permute(3,0,1,2)
    
    return my_noise_mat



def add_noise_list(wave_list,std):
    
    """add same noise level noise to the wavelet subbands"""
    
    Yl,Yh = wave_list
    device = Yl.device
    level = len(Yh)
    wave_mat = get_wave_mat(wave_list)

    is_complex = (Yh[0].shape[1] == 2)
    
    wave_mat_new = wave_mat.clone()
    
    wave_mat_new = wave_mat_new + generate_noise_mat(wave_mat_new.shape,std,is_complex,device)
    
    return wave_mat2list(wave_mat_new,level)




def add_noise_subbandwise_real_and_imag_list(wave_list,stds_real,stds_imag):
    
    """ add noise of different noise levels to each channel (i.e. real and imaginary channels) of each subband """
    
    Yl,Yh = wave_list
    device = Yl.device
    level = len(Yh)
    
    
    
    
    Yl_new = Yl_new + generate_noise_mat_real_and_imag(Yl.shape,stds_real[0],stds_imag[0],is_complex,device)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            Yh_new[index][:,:,j,:,:] = Yh_new[index][:,:,j,:,:] + generate_noise_mat_real_and_imag(Yh_new[index][:,:,j,:,:].shape,stds_real[count],stds_imag[count],is_complex,device) 
            count = count+1
    
    return [Yl_new,Yh_new]


def generate_noise_mat(my_shape,my_std,is_complex,device):
    
    if is_complex:
        my_std_new = my_std.clone()/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = my_std.clone()
    
    my_noise_mat = my_std_new*(torch.randn(my_shape, device = device))
    
    return my_noise_mat
    
def generate_noise_mat_real_and_imag(my_shape,my_std_real,my_std_imag,device):
    """second dimension is two channel representing the complex value"""

    my_noise_mat = torch.randn(my_shape, device = device)
    my_noise_mat[:,0,:,:] = my_std_real*my_noise_mat[:,0,:,:]
    my_noise_mat[:,1,:,:] = my_std_imag*my_noise_mat[:,1,:,:]
    
    return my_noise_mat

def add_noise_to_complex_measurements(y,wvar,idx1_complement,idx2_complement,device, is_complex):

    if is_complex:
        my_std_new = (torch.sqrt(wvar.clone()))/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = (torch.sqrt(wvar.clone()))
        
    noise = my_std_new*(torch.randn(y.shape, device = device))
    
    result = y.clone() + noise

    result[:,:,idx1_complement,idx2_complement] = 0
    
    ## Testing added noise

    noise[:,:,idx1_complement,idx2_complement] = 0
    
    pow_1 = torch.sum(y**2)
    pow_2 = torch.sum(noise**2)
    ratio_snr = torch.sqrt(pow_1)/torch.sqrt(pow_2)
    SNRdB_test = 20*torch.log10(ratio_snr)
    print('SNR in dB for this run:')
    print(SNRdB_test)
    
    ## Done Testing
    
    return result

def add_noise_to_complex_measurements_no_verbose(y,wvar,idx1_complement,idx2_complement,device, is_complex):

    if is_complex:
        my_std_new = (torch.sqrt(wvar.clone()))/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = (torch.sqrt(wvar.clone()))
        
    noise = my_std_new*(torch.randn(y.shape, device = device))
    
    result = y.clone() + noise

    result[:,:,idx1_complement,idx2_complement] = 0
    
    ## Testing added noise

    noise[:,:,idx1_complement,idx2_complement] = 0
    
    pow_1 = torch.sum(y**2)
    pow_2 = torch.sum(noise**2)
    ratio_snr = torch.sqrt(pow_1)/torch.sqrt(pow_2)
    SNRdB_test = 20*torch.log10(ratio_snr)

    
    ## Done Testing
    
    return result


def add_noise_to_complex_measurements_multi_coil(y,wvar,idx1_complement,idx2_complement,device, is_complex):

    if is_complex:
        my_std_new = (torch.sqrt(wvar.clone()))/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = (torch.sqrt(wvar.clone()))
        
    noise = my_std_new*(torch.randn(y.shape, device = device))
    
    result = y.clone() + noise

    result[:,:,idx1_complement,idx2_complement] = 0
    
    ## Testing added noise

    noise[:,:,idx1_complement,idx2_complement] = 0
    
    pow_1 = torch.sum(y**2)
    pow_2 = torch.sum(noise**2)
    ratio_snr = torch.sqrt(pow_1)/torch.sqrt(pow_2)
    SNRdB_test = 20*torch.log10(ratio_snr)
    print('SNR in dB for this run:')
    print(SNRdB_test)
    
    ## Done Testing
    
    return result





def get_p1m1_for_a_subband_with_size(zeros_wave_list,subband_num):

    Yl,Yh = zeros_wave_list
    device = Yl.device
    level = len(Yh)
    
    assert subband_num < 3*level+1, "subband_num is greater than number of available subbands"
    
    is_complex = (Yh[0].shape[1] == 2)
    
    if subband_num == 0:
        Yl = torch.sign(generate_noise_mat(Yl.shape,torch.tensor(1),is_complex,device))/torch.sqrt(torch.tensor(2))
        subband_size = torch.tensor(Yl.shape[-1]*Yl.shape[-2], device = device)
    else:
        count = 1
        for i in range(level):
            index = level-1-i
            for j in range(3):
                if count == subband_num:
                    Yh[index][:,:,j,:,:] = torch.sign(generate_noise_mat(Yh[index][:,:,j,:,:].shape,torch.tensor(1),is_complex,device))/torch.sqrt(torch.tensor(2))
                    subband_size = torch.tensor(Yh[index][:,:,j,:,:].shape[-1]*Yh[index][:,:,j,:,:].shape[-2], device = device)
                count = count+1
    
    return get_wave_mat([Yl,Yh]), subband_size


def get_p1m1_for_a_subband_with_size_2(zeros_wave_list,subband_num):

    Yl,Yh = zeros_wave_list
    device = Yl.device
    level = len(Yh)
    
    assert subband_num < 3*level+1, "subband_num is greater than number of available subbands"
    
    is_complex = (Yh[0].shape[1] == 2)
    
    if subband_num == 0:
        Yl = torch.sign(generate_noise_mat(Yl.shape,torch.tensor(1),is_complex,device))
        subband_size = torch.tensor(Yl.shape[-1]*Yl.shape[-2], device = device)
    else:
        count = 1
        for i in range(level):
            index = level-1-i
            for j in range(3):
                if count == subband_num:
                    Yh[index][:,:,j,:,:] = torch.sign(generate_noise_mat(Yh[index][:,:,j,:,:].shape,torch.tensor(1),is_complex,device))
                    subband_size = torch.tensor(Yh[index][:,:,j,:,:].shape[-1]*Yh[index][:,:,j,:,:].shape[-2], device = device)
                count = count+1
    
    return get_wave_mat([Yl,Yh]), subband_size


    
def get_my_jittred_batch(wave_mat,level,eta,p1m1_mask):
    """This is for processing multiple subband batches only
    """
    device = wave_mat.device
    jittred_batch = torch.zeros(3*level + 2, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)
    
    jittred_batch[:,:,:,:] = wave_mat.clone().repeat(3*level + 2,1,1,1)
    
    noise_mat = (((p1m1_mask*(torch.sign(torch.randn(3*level + 1, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)))))/torch.sqrt(torch.tensor(2,device = device)))
    
    jittred_batch[1:,:,:,:] = jittred_batch[1:,:,:,:] + eta*noise_mat

    return jittred_batch, noise_mat

# def get_my_jittred_batch_with_sens_map_mask(wave_mat,level,eta,p1m1_mask,sens_map_mask):
#     """This is for processing multiple subband batches only
#     """
#     device = wave_mat.device
#     jittred_batch = torch.zeros(3*level + 2, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)
    
#     jittred_batch[:,:,:,:] = wave_mat.clone().repeat(3*level + 2,1,1,1)
    
#     noise_mat = (((sens_map_mask*p1m1_mask*(torch.sign(torch.randn(3*level + 1, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)))))/torch.sqrt(torch.tensor(2,device = device)))
    
#     jittred_batch[1:,:,:,:] = jittred_batch[1:,:,:,:] + eta*noise_mat

#     return jittred_batch, noise_mat



def get_my_jittred_batch_2(wave_mat,level,eta_batch,p1m1_mask_batch):
    """This is for processing multiple image batches and also subband batches
    """
    device = wave_mat.device
    batch_len = wave_mat.shape[0]
    jittred_batch = torch.zeros(batch_len*(3*level + 2), wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)
    
    jittred_batch[0:batch_len,:,:,:] = wave_mat.clone()
    
    jittred_batch[batch_len:,:,:,:] = wave_mat.clone().repeat_interleave(3*level + 1, dim = 0)
    
    noise_mat = (((p1m1_mask_batch*(torch.sign(torch.randn(batch_len*(3*level + 1), wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)))))/torch.sqrt(torch.tensor(2,device = device)))
    
    jittred_batch[batch_len:,:,:,:] = jittred_batch[batch_len:,:,:,:] + (noise_mat.permute(1,2,3,0)*eta_batch).permute(3,0,1,2)

    return jittred_batch, noise_mat


# def get_my_jittred_batch(wave_mat,level,eta,p1m1_mask):
    
#     device = wave_mat.device
#     jittred_batch = torch.zeros(3*level + 2, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3]).to(device)
    
#     jittred_batch[:,:,:,:] = wave_mat.clone().repeat(3*level + 2,1,1,1)
    
#     jittred_batch[1:,:,:,:] = jittred_batch[1:,:,:,:] + ((eta*((p1m1_mask*(torch.sign(torch.randn(3*level + 1, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3])).to(device))).permute(1,2,3,0)))/torch.sqrt(torch.tensor(2))).permute(3,0,1,2)

#     return jittred_batch

def get_my_jittred_batch_subband(wave_mat,level,eta_subband,p1m1_mask):

    device = wave_mat.device
    jittred_batch = torch.zeros(3*level + 2, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)
    
    jittred_batch[:,:,:,:] = wave_mat.clone().repeat(3*level + 2,1,1,1)
    
    noise_mat = (((p1m1_mask*(torch.sign(torch.randn(3*level + 1, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)))))/torch.sqrt(torch.tensor(2,device = device)))
    
    jittred_batch[1:,:,:,:] = jittred_batch[1:,:,:,:] + (noise_mat.permute(1,2,3,0)*eta_subband).permute(3,0,1,2)

    return jittred_batch, noise_mat



def get_max_in_each_subband(wave_mat,p1m1_mask):
    
    device = wave_mat.device
    num_of_subbands = p1m1_mask.shape[0]
    
    wave_mat_batch = torch.zeros(num_of_subbands, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)
    
    wave_mat_batch[:,:,:,:] = wave_mat.clone().repeat(num_of_subbands,1,1,1)
    
    subband_mask = p1m1_mask*p1m1_mask
    
    wave_mat_split_batch = wave_mat_batch*subband_mask
    
    wave_mat_split_batch_abs = transforms_new.complex_abs(wave_mat_split_batch.permute(0,2,3,1))
    
    max_subbandwise = torch.amax(wave_mat_split_batch_abs, dim=(1,2))
    
    return max_subbandwise


def get_mean_in_each_subband(wave_mat,p1m1_mask,subband_sizes):
    
    device = wave_mat.device
    num_of_subbands = p1m1_mask.shape[0]
    
    wave_mat_batch = torch.zeros(num_of_subbands, wave_mat.shape[1],wave_mat.shape[2],wave_mat.shape[3], device = device)
    
    wave_mat_batch[:,:,:,:] = wave_mat.clone().repeat(num_of_subbands,1,1,1)
    
    subband_mask = p1m1_mask*p1m1_mask
    
    wave_mat_split_batch = wave_mat_batch*subband_mask
    
    wave_mat_split_batch_abs = transforms_new.complex_abs(wave_mat_split_batch.permute(0,2,3,1))
    
    mean_subbandwise = torch.sum(wave_mat_split_batch_abs, dim=(1,2))/subband_sizes
    
    return mean_subbandwise





def get_p1m1_mask_and_subband_sizes(zeros_wave_mat,level):
    
    device = zeros_wave_mat.device
    subband_sizes = torch.zeros(3*level + 1, device = device)
    p1m1_mask = torch.zeros(3*level + 1, zeros_wave_mat.shape[1],zeros_wave_mat.shape[2],zeros_wave_mat.shape[3], device = device)

    for i in range(3*level + 1):
        Yl,Yh = wave_mat2list(zeros_wave_mat.clone(),level)
        subband_num = i
        p1m1_mask[i,:,:,:], subband_sizes[i] = get_p1m1_for_a_subband_with_size_2([Yl,Yh], subband_num)
    
    return p1m1_mask, subband_sizes


def find_subband_wise_MSE_list_with_wave_mask(recov_wavelet_list, gt_list, wave_mask_list):
    
    Yl_r,Yh_r = recov_wavelet_list
    Yl,Yh = gt_list
    
    Yl_mask, Yh_mask = wave_mask_list
    
    level = len(Yh)
    
    mse_list = [None] * (3*level + 1)
    
    den = (torch.sum(Yl_mask>0)/2)
    
    mse_list[0] = (torch.sum((Yl_r*Yl_mask - Yl*Yl_mask)**2))/(den)
    
    count = 1
    for i in range(level):
        index = level-1-i
        den = (torch.sum(Yh_mask[index][:,:,0,:,:]>0)/2)
        for j in range(3):
            mse_list[count] = (torch.sum(((Yh_r[index][:,:,j,:,:])*(Yh_mask[index][:,:,j,:,:]) - (Yh[index][:,:,j,:,:])*(Yh_mask[index][:,:,j,:,:]))**2))/(den) 
            count = count+1
    
    return mse_list

def find_subband_wise_MSE_list_complex_with_wave_mask(recov_wavelet_list, gt_list, wave_mask_list):
    
    Yl_r,Yh_r = recov_wavelet_list
    Yl,Yh = gt_list
    Yl_mask, Yh_mask = wave_mask_list
    
    level = len(Yh)
    
    mse_list_real = [None] * (3*level + 1)
    mse_list_imag = [None] * (3*level + 1)
    
    den = (torch.sum(Yl_mask>0)/2)
    
    mse_list_real[0] = (torch.sum(((Yl_r[:,0,:,:])*(Yl_mask[:,0,:,:]) - (Yl[:,0,:,:])*(Yl_mask[:,0,:,:]))**2))/(den)
    mse_list_imag[0] = (torch.sum(((Yl_r[:,1,:,:])*(Yl_mask[:,1,:,:]) - (Yl[:,1,:,:])*(Yl_mask[:,1,:,:]))**2))/(den)
    
    count = 1
    for i in range(level):
        index = level-1-i
        den = (torch.sum(Yh_mask[index][:,:,0,:,:]>0)/2)
        for j in range(3):
            mse_list_real[count] = (torch.sum(((Yh_r[index][:,0,j,:,:])*(Yh_mask[index][:,0,j,:,:]) - (Yh[index][:,0,j,:,:])*(Yh_mask[index][:,0,j,:,:]))**2))/(den) 
            mse_list_imag[count] = (torch.sum(((Yh_r[index][:,1,j,:,:])*(Yh_mask[index][:,1,j,:,:]) - (Yh[index][:,1,j,:,:])*(Yh_mask[index][:,1,j,:,:]))**2))/(den) 
            count = count+1
    
    return mse_list_real,mse_list_imag





def find_subband_wise_MSE_list(recov_wavelet_list, gt_list):
    
    Yl_r,Yh_r = recov_wavelet_list
    Yl,Yh = gt_list
    
    level = len(Yh)
    
    mse_list = [None] * (3*level + 1)
    
    mse_list[0] = (torch.sum((Yl_r - Yl)**2))/(Yl.shape[-1]**2)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            mse_list[count] = (torch.sum((Yh_r[index][:,:,j,:,:] - Yh[index][:,:,j,:,:])**2))/(Yh[index][:,:,j,:,:].shape[-1]**2) 
            count = count+1
    
    return mse_list

def find_subband_wise_MSE_list_complex(recov_wavelet_list, gt_list):
    
    Yl_r,Yh_r = recov_wavelet_list
    Yl,Yh = gt_list
    
    level = len(Yh)
    
    mse_list_real = [None] * (3*level + 1)
    mse_list_imag = [None] * (3*level + 1)
    
    mse_list_real[0] = (torch.sum((Yl_r[:,0,:,:] - Yl[:,0,:,:])**2))/(Yl.shape[-1]**2)
    mse_list_imag[0] = (torch.sum((Yl_r[:,1,:,:] - Yl[:,1,:,:])**2))/(Yl.shape[-1]**2)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            mse_list_real[count] = (torch.sum((Yh_r[index][:,0,j,:,:] - Yh[index][:,0,j,:,:])**2))/(Yh[index][:,:,j,:,:].shape[-1]**2) 
            mse_list_imag[count] = (torch.sum((Yh_r[index][:,1,j,:,:] - Yh[index][:,1,j,:,:])**2))/(Yh[index][:,:,j,:,:].shape[-1]**2) 
            count = count+1
    
    return mse_list_real,mse_list_imag





def find_subband_wise_MSE_list_batch(recov_wavelet_list, gt_list):
    
    Yl_r,Yh_r = recov_wavelet_list
    Yl,Yh = gt_list
    
    level = len(Yh)
    
    mse_list = torch.zeros(Yl_r.shape[0],(3*level + 1),device = Yl_r.device)
    
    mse_list[:,0] = (torch.sum((Yl_r - Yl)**2, (1,2,3)))/(Yl.shape[-1]**2)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            mse_list[:,count] = (torch.sum((Yh_r[index][:,:,j,:,:] - Yh[index][:,:,j,:,:])**2, (1,2,3)))/(Yh[index][:,:,j,:,:].shape[-1]**2) 
            count = count+1
    
    return mse_list