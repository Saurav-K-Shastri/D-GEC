def CG_method_warm_strt_2(b,OP_A,x, r, p, CG_flag, max_iter,eps_lim):
    
    if CG_flag == False:
#         print("hello")
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
        
    print(torch.sqrt(rsnew))
    return x, r, p, i, torch.sqrt(rsnew)


def calc_MC_divergence_complex_with_warm_strt_2(denoiser, denoised, wavelet_mat, variances, x_init_MC, r_init_MC, p_init_MC, CG_flag, level):

    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach. This code uses warm start
    """
    device = wavelet_mat.device
    
    next_x_warm_MC = torch.zeros_like(x_init_MC)
    next_r_CG_warm_MC = torch.zeros_like(r_init_MC)
    next_p_CG_warm_MC = torch.zeros_like(p_init_MC)
    
    Yl,Yh = wutils.wave_mat2list(wavelet_mat,level)
    
    alpha = [None] * (3*level + 1)
    wavelet_mat_jittered = wavelet_mat.clone()
    eta1 = torch.max(transforms_new.complex_abs(Yl.permute(0,2,3,1)))/1000.
    eta2 = torch.mean(torch.sqrt(variances))
    
    eta = torch.max(eta1,eta2)
    eps = torch.tensor(2.22e-16).to(device)
    eta = eta + eps 
    
    noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),0)

    
    wavelet_mat_jittered += eta * noise_vec
    
    denoised_jittered, r_CG, p_CG = denoiser(wavelet_mat_jittered, variances, x_init_MC[0,:,:,:].unsqueeze(0), r_init_MC[0,:,:,:].unsqueeze(0), p_init_MC[0,:,:,:].unsqueeze(0), CG_flag)
    
    next_x_warm_MC[0,:,:,:] = (denoised_jittered.squeeze(0)).clone()
    next_r_CG_warm_MC[0,:,:,:] = (r_CG.squeeze(0)).clone()
    next_p_CG_warm_MC[0,:,:,:] = (p_CG.squeeze(0)).clone()
    
    alpha[0] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(((denoised_jittered - denoised)/eta).permute(0,2,3,1).contiguous()).reshape(-1))))

    

    count = 1
    for s in range(level):
        index = level - 1 - s
        for b in range(3):
            
            wavelet_mat_jittered = (wavelet_mat).clone()
            eta1 = torch.max(transforms_new.complex_abs(Yh[index][:,:,b,:,:].permute(0,2,3,1)))/1000.
            eta = torch.max(eta1,eta2)
            eps = torch.tensor(2.22e-16).to(device) 
            eta = eta + eps
            
            noise_vec, subband_size = wutils.get_p1m1_for_a_subband_with_size(wutils.wave_mat2list(torch.zeros_like(wavelet_mat),level),count)

            wavelet_mat_jittered += eta * noise_vec

            denoised_jittered, r_CG, p_CG = denoiser(wavelet_mat_jittered, variances, x_init_MC[count,:,:,:].unsqueeze(0), r_init_MC[count,:,:,:].unsqueeze(0), p_init_MC[count,:,:,:].unsqueeze(0), CG_flag)
            next_x_warm_MC[count,:,:,:] = (denoised_jittered.squeeze(0)).clone()
            next_r_CG_warm_MC[count,:,:,:] = (r_CG.squeeze(0)).clone()
            next_p_CG_warm_MC[count,:,:,:] = (p_CG.squeeze(0)).clone()
            
            alpha[count] = (1. / subband_size)*(torch.real(torch.dot(torch.conj(torch.view_as_complex(noise_vec.permute(0,2,3,1).contiguous()).reshape(-1)),torch.view_as_complex(((denoised_jittered - denoised)/eta).permute(0,2,3,1).contiguous()).reshape(-1))))

            count = count + 1 
                    
    return alpha, next_x_warm_MC, next_r_CG_warm_MC, next_p_CG_warm_MC


class LMSE_CG_GEC_with_div_and_warm_strt_2:
    """Wrapper of LMSE for using with GEC which used CG. This code uses warm start"""
    def __init__(self, y, idx1_complement, idx2_complement,sigma_w,level = 4,LMSE_inner_iter_lim = 100, beta_tune_LMSE = torch.tensor(1), eps_lim = torch.tensor(1e-4)):

        self.y = y
        self.idx1_complement = idx1_complement
        self.idx2_complement = idx2_complement
        self.sigma_w = sigma_w
        self.level = level
        self.LMSE_inner_iter_lim = LMSE_inner_iter_lim
        self.beta_tune_LMSE = beta_tune_LMSE
        self.eps_lim = eps_lim
        
    def __call__(self, wavelet_mat, variances, x_init, x_init_MC, r_init, r_init_MC, p_init, p_init_MC, CG_flag, calc_divergence=True):

        variances *= self.beta_tune_LMSE
       
        denoised, next_r_warm, next_p_warm  = self._denoise(wavelet_mat, variances, x_init, r_init, p_init, CG_flag)
        if calc_divergence:
            alpha, next_x_warm_MC, next_r_CG_warm_MC, next_p_CG_warm_MC  = calc_MC_divergence_complex_with_warm_strt_2(self._denoise, denoised, wavelet_mat, variances, x_init_MC, r_init_MC, p_init_MC, CG_flag, self.level)
            return denoised, alpha , next_r_warm, next_p_warm, next_x_warm_MC, next_r_CG_warm_MC, next_p_CG_warm_MC
        else:
            return denoised

    def _denoise(self, wavelet_mat, variances, x_init, r_init, p_init, CG_flag):
        
        GAMMA_1_full = 1/variances
        my_scalar_vec = (self.sigma_w**2)*(GAMMA_1_full)
        
        B_op_foo = B_op(self.idx1_complement,self.idx2_complement)       
        A_bar = CG_op(B_op_foo,self.level,my_scalar_vec)
        
        CG_INP_op_foo = CG_INP_op(B_op_foo,self.level,my_scalar_vec) 
        b_bar = CG_INP_op_foo.forward(self.y,wavelet_mat)
        
#         x_init = torch.zeros_like(wavelet_mat)
#         x_init = torch.randn_like(wavelet_mat)
#         x_init = B_op_foo.H(self.y)
        
        denoised_wavelet, r_CG, p_CG, stop_iter,rtr_end = CG_method_warm_strt_2(b_bar,A_bar,x_init, r_init, p_init, CG_flag ,self.LMSE_inner_iter_lim,self.eps_lim) 

        return denoised_wavelet, r_CG, p_CG


    
    