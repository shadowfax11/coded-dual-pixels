import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import utils.pytorch_ssim as pytorch_ssim
import torchmetrics
from scipy.stats import rankdata
import lpips

# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def get_score(pred_dict, gt_dict, depth_score_type, aif_score_type):
    score_dict = {}
    if 'depth' in pred_dict.keys():
        if 'rmse' in depth_score_type:
            score_dict['depth_rmse'] = torch.sqrt(get_mse_loss(pred_dict['depth'], gt_dict['depth']))
        if 'mse' in depth_score_type:
            score_dict['depth_mse'] = get_mse_loss(pred_dict['depth'], gt_dict['depth'])
        if 'absrel' in depth_score_type:
            score_dict['depth_absrel'] = get_absrel_loss(pred_dict['depth'], gt_dict['depth'])
        if 'mae' in depth_score_type:
            score_dict['depth_mae'] = get_mae_loss(pred_dict['depth'], gt_dict['depth'])
        if 'delta1' in depth_score_type:
            score_dict['depth_delta1'] = get_delta_loss(pred_dict['depth'], gt_dict['depth'], 1)
        if 'delta2' in depth_score_type:
            score_dict['depth_delta2'] = get_delta_loss(pred_dict['depth'], gt_dict['depth'], 2)
        if 'delta3' in depth_score_type:
            score_dict['depth_delta3'] = get_delta_loss(pred_dict['depth'], gt_dict['depth'], 3)
        if 'ai1' in depth_score_type: 
            pred_disp = pred_dict['defocus'].cpu().numpy()
            gt_disp = gt_dict['defocus'].cpu().numpy()
            score_dict['depth_ai1'], _ = affine_invariant_1(pred_disp, gt_disp)
        if 'ai1' in depth_score_type: 
            pred_disp = pred_dict['defocus'].cpu().numpy()
            gt_disp = gt_dict['defocus'].cpu().numpy()
            score_dict['depth_ai2'], _ = affine_invariant_2(pred_disp, gt_disp)
        if 'sps' in depth_score_type:
            pred_disp = pred_dict['defocus'].cpu().numpy()
            gt_disp = gt_dict['defocus'].cpu().numpy()
            _, b2 = affine_invariant_2(pred_disp, gt_disp)
            pred_disp_affine = pred_disp*b2[0] + b2[1]
            score_dict['depth_sps'] = 1 - np.abs(spearman_correlation(pred_disp_affine, gt_disp))
    if 'aif' in pred_dict.keys():
        if 'mse' in aif_score_type:
            score_dict['aif_mse'] = get_mse_loss(pred_dict['aif'], gt_dict['aif'])
        if 'psnr' in aif_score_type:
            score_dict['aif_psnr'] = 20*torch.log10(1./torch.sqrt(get_mse_loss(pred_dict['aif'], gt_dict['aif'])))
        if 'ssim' in aif_score_type: 
            score_dict['aif_ssim'] = pytorch_ssim.ssim(pred_dict['aif'], gt_dict['aif'])
        if 'lpips' in aif_score_type: 
            score_dict['aif_lpips'] = get_lpips_score(pred_dict['aif'], gt_dict['aif'])
    return score_dict

def get_loss(pred_dict, gt_dict, defocus_loss_type, depth_loss_type, aif_loss_type, defocus_loss_wt=1.0, depth_loss_wt=1.0, aif_loss_wt=1.0, args=None):
    """
    Adds all losses to the loss functions list. Includes regularizers as well. Relative strengths have been determined heuristically. 
    choices = ["mse","grad","vgg","ssim"]
    """
    loss_dict = {}
    if 'defocus' in pred_dict.keys():
        if 'l1' in defocus_loss_type: 
            loss_dict['defocus_l1'] = defocus_loss_wt * get_l1_loss(pred_dict['defocus'], gt_dict['defocus'])
        if 'mse' in defocus_loss_type: 
            loss_dict['defocus_mse'] = defocus_loss_wt * get_mse_loss(pred_dict['defocus'], gt_dict['defocus'])
        if 'grad' in defocus_loss_type:
            loss_dict['defocus_grad'] = 0.5 * defocus_loss_wt * get_grad_loss_v2(pred_dict['defocus'], gt_dict['defocus'])
            # loss_dict['defocus_grad'] = 10 * defocus_loss_wt * get_grad_loss(pred_dict['defocus'], gt_dict['defocus'])
        if 'ssim' in defocus_loss_type:
            loss_dict['defocus_ssim'] = 0.01 * defocus_loss_wt * (1. - pytorch_ssim.ssim(pred_dict['defocus'], gt_dict['defocus']))/2.
        if 'smoothl1' in defocus_loss_type:
            loss_dict['defocus_smoothl1'] = defocus_loss_wt * F.smooth_l1_loss(pred_dict['defocus'], gt_dict['defocus'], reduction='mean', beta=0.001)
        if 'reg_tv-l1' in defocus_loss_type: 
            loss_dict['defocus_reg_tv-l1'] = 5e-6 * defocus_loss_wt * TV2DNorm('l1')(pred_dict['defocus'])
        if 'reg_tv-hessian' in defocus_loss_type: 
            loss_dict['reg_tv-hessian'] = 5e-6 * defocus_loss_wt * TV2DNorm('hessian')(pred_dict['defocus'])
        if 'reg_tv-isotropic' in defocus_loss_type:
            loss_dict['reg_tv-isotropic'] = 5e-6 * defocus_loss_wt * TV2DNorm('isotropic')(pred_dict['defocus'])
    if 'depth' in pred_dict.keys():
        if 'mse' in depth_loss_type:
            loss_dict['depth_mse'] = depth_loss_wt * get_mse_loss(pred_dict['depth'], gt_dict['depth'])
        if 'grad' in depth_loss_type: 
            loss_dict['depth_grad'] = depth_loss_wt * get_grad_loss(pred_dict['depth'], gt_dict['depth'])
        if 'ssim_inv' in depth_loss_type:
            pred_depth_inv = args.max_depth_mm/pred_dict['depth']
            gt_depth_inv = args.max_depth_mm/gt_dict['depth']
            loss_dict['depth_ssim_inv'] = 1000 * depth_loss_wt * (1.-pytorch_ssim.ssim(pred_depth_inv,gt_depth_inv))/2.
        if 'ssim' in depth_loss_type: 
            raise NotImplementedError
    if 'aif' in pred_dict.keys():
        if 'l1' in aif_loss_type: 
            loss_dict['aif_l1'] = aif_loss_wt * get_l1_loss(pred_dict['aif'], gt_dict['aif'])
        if 'mse' in aif_loss_type:
            loss_dict['aif_mse'] = aif_loss_wt * get_mse_loss(pred_dict['aif'], gt_dict['aif'])
        if 'grad' in aif_loss_type:
            loss_dict['aif_grad'] = 0.5 * aif_loss_wt * get_grad_loss_v2(pred_dict['aif'], gt_dict['aif'])
            # loss_dict['aif_grad'] = aif_loss_wt * get_grad_loss(pred_dict['aif'], gt_dict['aif'])
        if 'ssim' in aif_loss_type: 
            loss_dict['aif_ssim'] = 0.01 * aif_loss_wt * (1.-pytorch_ssim.ssim(pred_dict['aif'], gt_dict['aif']))/2.
        if 'reg_tv-l1' in aif_loss_type: 
            loss_dict['aif_reg_tv-l1'] = 5e-6 * aif_loss_wt * TV2DNorm('l1')(pred_dict['aif'])
        if 'reg_tv-hessian' in aif_loss_type: 
            loss_dict['reg_tv-hessian'] = 5e-6 * aif_loss_wt * TV2DNorm('hessian')(pred_dict['aif'])
        if 'reg_tv-isotropic' in aif_loss_type:
            loss_dict['reg_tv-isotropic'] = 5e-6 * aif_loss_wt * TV2DNorm('isotropic')(pred_dict['aif'])
    
    loss = 0.
    for k, v in loss_dict.items():
        loss += v

    return loss, loss_dict

def get_lpips_score(x1,x2):
    lpips_fn = lpips.LPIPS()
    if x1.shape[1]==1: x1 = x1.repeat(1,3,1,1)
    if x2.shape[1]==1: x2 = x2.repeat(1,3,1,1)
    x1 = torch.clamp(2*(x1-0.5), min=-1, max=+1)
    x2 = torch.clamp(2*(x2-0.5), min=-1, max=+1)
    return lpips_fn.forward(x1,x2) 

def compute_gradient2d_maps(f):
    fx = f[...,:,1:] - f[...,:,:-1]
    fy = f[...,1:,:] - f[...,:-1,:]
    return fx[...,:-1,:], fy[...,:,:-1]

def get_grad_loss_v2(pred, target):
    y_grad_true, x_grad_true = torchmetrics.functional.image_gradients(target)
    y_grad_pred, x_grad_pred = torchmetrics.functional.image_gradients(pred)
    grad_loss = F.mse_loss(y_grad_pred, y_grad_true) + F.mse_loss(x_grad_pred, x_grad_true)
    return grad_loss

def get_grad_loss(x1, x2):
    x1_dx, x1_dy = compute_gradient2d_maps(x1)
    x2_dx, x2_dy = compute_gradient2d_maps(x2)
    grad_value = torch.mean((x1_dx-x2_dx)**2 + (x1_dy-x2_dy)**2)
    return grad_value

def get_l1_loss(x1, x2):
    l1_value = torch.mean(torch.abs(x1-x2))
    return l1_value

def get_mse_loss(x1, x2):
    mse_value = torch.mean((x1-x2)**2)
    return mse_value

def get_absrel_loss(x1, x2):
    absrel_value = torch.mean(torch.abs((x2-x1)/x2))
    return absrel_value

def get_mae_loss(x1, x2):
    mae_value = torch.mean(torch.abs(x2-x1))
    return mae_value

def get_delta_loss(x1, x2, i):
    delta_value = torch.mean((torch.max(x2/x1, x1/x2) < 1.05**i).float())
    return delta_value


class Hessian2DNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[..., 1:-1, :-2] + img[..., 1:-1, 2:] - 2*img[..., 1:-1, 1:-1]
        fyy = img[..., :-2, 1:-1] + img[..., 2:, 1:-1] - 2*img[..., 1:-1, 1:-1]
        fxy = img[..., :-1, :-1] + img[..., 1:, 1:] - \
              img[..., 1:, :-1] - img[..., :-1, 1:]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2)).sum()

class Hessian3DNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[...,1:-1, 1:-1, :-2] + img[...,1:-1, 1:-1, 2:] - 2*img[...,1:-1, 1:-1, 1:-1]
        fyy = img[...,1:-1, :-2, 1:-1] + img[...,1:-1, 2:, 1:-1] - 2*img[...,1:-1, 1:-1, 1:-1]
        fxy = img[...,1:-1, :-1, :-1] + img[...,1:-1, 1:, 1:] - \
                img[...,1:-1, 1:, :-1] - img[...,1:-1, :-1, 1:]
        fzz = img[...,:-2, 1:-1, 1:-1] + img[...,2:, 1:-1, 1:-1] - 2*img[...,1:-1, 1:-1, 1:-1]
        fxz = img[...,:-1, 1:-1, :-1] + img[...,1:, 1:-1, 1:] - \
                img[...,1:, 1:-1, :-1] - img[...,:-1, 1:-1, 1:]
        fyz = img[...,:-1, :-1, 1:-1] + img[...,1:, 1:, 1:-1] - \
                img[...,1:, :-1, 1:-1] - img[...,:-1, 1:, 1:-1]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2) + fzz.abs().pow(2) +\
                          2*fxz[...,:-1, :, :-1].abs().pow(2) + 2*fyz[...,:-1,:-1,:].abs().pow(2) ).sum()

class TV2DNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[..., 1:, 1:] - img[..., 1:, :-1]
        grad_y = img[..., 1:, 1:] - img[..., :-1, 1:]
        
        if self.mode == 'isotropic':
            #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
            return torch.sqrt(grad_x**2 + grad_y**2).sum()
        elif self.mode == 'l1':
            return abs(grad_x).sum() + abs(grad_y).sum()
        elif self.mode == 'hessian':
            return Hessian2DNorm()(img)
        else:
            return (grad_x.pow(2) + grad_y.pow(2)).sum()     
       
class TV3DNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[...,1:, 1:, 1:] - img[...,1:, 1:, :-1]
        grad_y = img[...,1:, 1:, 1:] - img[...,1:, :-1, 1:]
        grad_z = img[...,1:, 1:, 1:] - img[...,:-1, 1:, 1:]
        
        if self.mode == 'isotropic':
            #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
            return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2).sum()
        elif self.mode == 'l1':
            return abs(grad_x).sum() + abs(grad_y).sum() + abs(grad_z).sum() 
        elif self.mode == 'hessian':
            return Hessian3DNorm()(img)
        else:
            return (grad_x.pow(2) + grad_y.pow(2) + grad_z.pow(2)).sum()     

'''
Following metrics originally from Garg et al., ICCV 2019,
https://github.com/google-research/google-research/blob/master/dual_pixels/eval/get_metrics.py
under Apache License, Version 2.0
'''
def affine_invariant_1(
        Y: np.ndarray,
        Target: np.ndarray,
        confidence_map = None,
        irls_iters = 5,
        eps = 1e-3):
    assert Y.shape==Target.shape
    if confidence_map is None: confidence_map = np.ones_like(Target)
    y = Y.ravel() # [N,]
    t = Target.ravel() # [N,]
    conf = confidence_map.ravel() # [N,]

    # w : IRLS weight
    # b : affine parameter
    # initialize IRLS weight
    w = np.ones_like(y, float) # [N,]
    ones = np.ones_like(y, float)
    # run IRLS
    for _ in range(irls_iters):
        w_sqrt = np.sqrt(w * conf) # [N,]
        WX = w_sqrt[:, None] * np.stack([y, ones], 1) # [N,1] * [N,2] = [N,2] (broadcast)
        Wt = w_sqrt * t # [N,]
        # solve linear system: WXb - Wt
        b = np.linalg.lstsq(WX, Wt, rcond=None)[0] # [2,]
        affine_y = y * b[0] + b[1]
        residual = np.abs(affine_y - t)
        # re-compute weight with clipping residuals
        w = 1 / np.maximum(eps, residual)
    
    # finally,
    ai1 = np.sum(conf * residual) / np.sum(conf)
    return ai1, b


def affine_invariant_2(
        Y: np.ndarray,
        Target: np.ndarray,
        confidence_map = None,
        eps = 1e-3):
    assert Y.shape==Target.shape
    if confidence_map is None: confidence_map = np.ones_like(Target)
    y = Y.ravel() # [N,]
    t = Target.ravel() # [N,]
    conf = confidence_map.ravel() # [N,]

    ones = np.ones_like(y, float)
    X = conf[:, None] * np.stack([y, ones], 1) # [N,1] * [N,2] = [N,2] (broadcast)
    t = conf * t # [N,]
    b = np.linalg.lstsq(X, t, rcond=None)[0] # [2,]
    affine_y = y * b[0] + b[1]

    # clipping residuals
    residual_sq = np.minimum(np.square(affine_y - t), np.finfo(np.float32).max)

    # finally,
    ai2 = np.sqrt(np.sum(conf * residual_sq) / np.sum(conf))
    return ai2, b

def spearman_correlation(
        X: np.ndarray,
        Y: np.ndarray,
        W = None):
    assert X.shape == Y.shape
    if W is None: W = np.ones_like(X)
    x, y, w = X.ravel(), Y.ravel(), W.ravel()

    # scale rank to -1 to 1 (for numerical stability)
    def _rescale_rank(z): return (z - len(z) // 2) / (len(z) // 2)
    rx = _rescale_rank(rankdata(x, method='dense'))
    ry = _rescale_rank(rankdata(y, method='dense'))

    def E(z): return np.sum(w * z) / np.sum(w)
    def _pearson_correlation(x, y):
        mu_x = E(x)
        mu_y = E(y)
        var_x = E(x * x) - mu_x * mu_x
        var_y = E(y * y) - mu_y * mu_y
        return (E(x * y) - mu_x * mu_y) / (np.sqrt(var_x * var_y))

    return _pearson_correlation(rx, ry)