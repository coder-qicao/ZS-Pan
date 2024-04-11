# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/4/14 21:03
# @Author  : Xiao Wu

from Toolbox.wald_utilities import MTF
#from mtf import MTF_MS
import torch
from torch.nn import functional as F
import numpy as np


def norm(tensor, order=2, axis=None):
    """Computes the l-`order` norm of a tensor.

    Parameters
    ----------
    tensor : tl.tensor
    order : int
    axis : int or tuple

    Returns
    -------
    float or tensor
        If `axis` is provided returns a tensor.
    """
    # handle difference in default axis notation
    if axis == ():
        axis = None

    if order == 'inf':
        return torch.max(torch.abs(tensor), dim=axis)
    if order == 1:
        return torch.sum(torch.abs(tensor), dim=axis)
    elif order == 2:
        return torch.sqrt(torch.sum(torch.abs(tensor) ** 2, dim=axis))
    else:
        return torch.sum(torch.abs(tensor) ** order, dim=axis) ** (1 / order)


def indexes_evaluation_fs(sr, lrms, ms, pan, L, th_values, sensor='none', ratio=6, mode='QS'):
    if th_values:
        sr[sr > 2 ** L] = 2 ** L
        sr[sr < 0] = 0

    if mode == 'QS':
        QNR_index, D_lambda, D_S = QS(sr, ms, lrms, pan, ratio)
        return QNR_index, D_lambda, D_S


def img_ssim(img1, img2, block_size):
    img1 = img1.float()
    img2 = img2.float()

    _, channel, h, w = img1.size()
    N = block_size ** 2
    sum2filter = torch.ones([channel, 1, block_size, block_size]).cuda()
    # print(img1.shape, sum2filter.shape)
    img1_sum = F.conv2d(img1, sum2filter, groups=channel)
    img2_sum = F.conv2d(img2, sum2filter, groups=channel)
    img1_sq_sum = F.conv2d(img1 * img1, sum2filter, groups=channel)
    img2_sq_sum = F.conv2d(img2 * img2, sum2filter, groups=channel)
    img12_sum = F.conv2d(img1 * img2, sum2filter, groups=channel)
    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = torch.ones_like(denominator)
    two = 2 * torch.ones_like(denominator)
    zeros = torch.zeros_like(denominator)
    # zeros_2 = torch.zeros_like(img12_sq_sum_mul)
    # index = (denominator1 == 0) and (img12_sq_sum_mul != 0)
    # quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]

    quality_map = torch.where((denominator1 == zeros).float() + (img12_sq_sum_mul != zeros).float() == two,
                              2 * img12_sum_mul / img12_sq_sum_mul, quality_map)
    # index = denominator != 0
    # quality_map[index] = numerator[index] / denominator[index]
    quality_map = torch.where(denominator != zeros, numerator / denominator, quality_map)


    return quality_map.mean(2).mean(2)


def Qavg(im1, im2, S):
    Q_orig = img_ssim(im1, im2, S)
    Q_avg = torch.mean(Q_orig)

    return Q_avg


def D_lambda_k(sr, ms, ratio, sensor, S):
    assert sr.shape == ms.shape, print("ms shape is not equal to sr shape")

    H, W, nbands = sr.shape

    if H % S != 0 or W % S != 0:
        raise ValueError("H, W must be multiple of the block size")

    fused_degraded = MTF(sr, sensor, ratio, nbands)

    ms = ms.permute(2, 0, 1).unsqueeze(0)
    Q2n_index = Qavg(ms, fused_degraded, S)
    Dl = 1 - Q2n_index

    return Dl


def compute_regress(target, input, alpha=0.05):
    # Multiple linear regression using least squares
    # alpha 跟 R2 没有关系
    n, channel = input.shape

    if target.shape[0] != n:
        raise ValueError(f'InvalidData, {target.shape[0]} != {n}')

    if torch.isnan(input).int().sum():
        raise ValueError(f'NaN in input')
    '''
    Q: -0.0141
    R: -21.2489
    '''
    #Q, R = torch.qr(input)  # 4096, 4096, 4096, 69
    Q, R = torch.linalg.qr(input, 'reduced' if True else 'complete')
    # perm = torch.argsort(torch.abs(torch.diag(R)))
    # Q, R = torch.qr(input[:, perm])

    # 少另外两个空条件
    # p = sum(abs(torch.diag(R)) > max(n, channel) * torch.ones(R[0]) * 1e-6)
    LS_coef = torch.inverse(R) @ (Q.T @ target)  # 0.8853, ...
    # LS_coef = LS_coef[perm]
    # RI = torch.reciprocal(torch.eye(int(p)) / R)
    # nu = max(0, n-p) # 5759931
    target_hat = input @ LS_coef  # 4096, 1
    r = target - target_hat
    norm_r = norm(r, axis=(0, 1))  # 0.0763

    # if nu != 0:
    #     rmse = norm_r / np.sqrt(nu)
    # tval = tinv((1-alpha/2), nu)

    SSE = norm_r * norm_r
    TSS = torch.pow(norm(target - torch.mean(target), axis=(0, 1)), 2)
    r2 = 1 - SSE / TSS
    # print("QR grad: ", r2.requires_grad)

    return r2


def QS(sr, ms, lrms, pan, ratio, sensor='none',
       beta=1, alpha=1, q=1, p=1, S=30):
    Flat_P = torch.reshape(pan, [-1, 1])
    Flat_F = torch.reshape(sr, (-1, sr.shape[-1])) / (2**16 - 1)
    Flat_P = Flat_P / (2**16 - 1)

    # print(torch.mean(Flat_F), torch.mean(Flat_P))
    # print(sr.shape)
    # print(sr[:3, :3, 0])

    R_square = compute_regress(Flat_P, Flat_F)
    # print("R_square:", R_square)
    D_s_index = 1 - R_square
    '''
    SR
    3.2917    3.2747    3.2591    3.2281    3.1628
    3.2935    3.2831    3.2593    3.2276    3.1706
    3.2681    3.2634    3.2501    3.2227    3.1730
    3.2385    3.2351    3.2248    3.2083    3.1706
    3.2103    3.2003    3.1963    3.1829    3.1513
    
    
    I_MS
    3.2625    3.2517    3.2301    3.1955    3.1379
    3.2565    3.2460    3.2251    3.1914    3.1354
    3.2445    3.2347    3.2149    3.1833    3.1305
    3.2264    3.2176    3.1998    3.1714    3.1240
    3.2013    3.1942    3.1800    3.1572    3.1192
    '''
    D_lambda_index = D_lambda_k(sr.transpose(1, 0), ms.transpose(1, 0), ratio, sensor, S)
    QNR_index = (1 - D_lambda_index) ** alpha * (1 - D_s_index) ** beta

    return QNR_index, D_lambda_index, D_s_index


if __name__ == '__main__':
    #import h5py
    import scipy.io as sio
    import os

    mat_path = "D:/Python/matlab_code/pansharpening/Pansharpening Toolbox for Distribution/hsp"

    FR = sio.loadmat(os.path.join(mat_path, 'DatasetFR1_lr.mat'))
    I_GS = sio.loadmat(os.path.join(mat_path, 'I_GS.mat'))['I_GS'].transpose(1, 0, 2) / 65535.0
    # I_GS = np.array(h5py.File(os.path.join(mat_path, 'I_GS.mat'), 'r')['I_GS']).transpose(2, 1, 0) / 65535.0

    print(I_GS[:3, :3, 0])

    I_MS_LR = FR['I_MS'].transpose(1, 0, 2) / 65535.0
    I_PAN = FR['I_PAN'].transpose(1, 0) / 65535.0
    print(I_GS.dtype, 2 ** 16 - 1)
    print(I_GS.shape, type(I_GS), I_MS_LR.shape, type(I_MS_LR), I_PAN.shape, type(I_PAN))
    # from_numpy仅支持float

    # sio.savemat("I_GS.mat", {'I_GS': I_GS[:64, :64]})

    I_GS, I_MS_LR, I_PAN = torch.from_numpy(I_GS) * 65535, torch.from_numpy(I_MS_LR) * 65535, torch.from_numpy(
        I_PAN) * 65535
    I_GS = torch.tensor(I_GS, requires_grad=True)
    print(I_GS.requires_grad)
    QNR_index, D_lambda_index, D_s_index = QS(I_GS, I_MS_LR, None, I_PAN, ratio=6, S=4)
    print(QNR_index, D_lambda_index, D_s_index)
