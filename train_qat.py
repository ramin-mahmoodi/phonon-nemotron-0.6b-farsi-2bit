"""
Nemotron 0.6B Farsi 2-Bit Industrial QAT Training Script
Continuous Straight-Through Estimator (STE) with Dual Distillation and SpecAugment
Author: Ramin Mahmoodi (https://github.com/ramin-mahmoodi)
License: Apache-2.0
"""

import os
import gc
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForRNNT, AutoProcessor
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

GROUP_SIZE = 128
MODEL_ID = "mehdi-hf/nemotron-asr-streaming-farsi"


class SparseFiveValueSTELight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, group_size=128):
        orig_shape = w.shape
        flat = w.reshape(-1, group_size)
        abs_w = flat.abs()
        sigma = abs_w.mean(dim=-1, keepdim=True)
        t0 = 0.35 * sigma
        t1 = 1.15 * sigma

        mask_lo = (abs_w >= t0) & (abs_w < t1)
        mask_hi = abs_w >= t1

        cnt_lo = mask_lo.sum(dim=-1, keepdim=True).clamp(min=1)
        cnt_hi = mask_hi.sum(dim=-1, keepdim=True).clamp(min=1)

        s_lo = (torch.where(mask_lo, abs_w, torch.zeros_like(abs_w)).sum(dim=-1, keepdim=True) / cnt_lo).clamp(min=1e-5)
        s_hi = (torch.where(mask_hi, abs_w, torch.zeros_like(abs_w)).sum(dim=-1, keepdim=True) / cnt_hi).clamp(min=1e-5)

        sign = torch.sign(flat)
        is_zero = abs_w < t0
        sign[is_zero] = 0.0
        mag = torch.where(mask_hi, s_hi, s_lo)

        ctx.save_for_backward(flat, is_zero)
        return (sign * mag).reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        flat, is_zero = ctx.saved_tensors
        grad_w = grad_output.clone()
        grad_flat = grad_w.reshape(-1, flat.shape[-1])
        grad_flat = torch.where(is_zero, grad_flat * 0.1, grad_flat)
        return grad_flat.reshape(grad_w.shape), None


def make_dynamic_linear(orig_linear):
    orig_forward = orig_linear.forward
    def custom_forward(x):
        qw = SparseFiveValueSTELight.apply(orig_linear.weight, GROUP_SIZE)
        return F.linear(x, qw, orig_linear.bias)
    return custom_forward


def apply_spec_augment(features, freq_mask_param=12, time_mask_param=25, num_freq_masks=2, num_time_masks=2):
    augmented = features.clone()
    if augmented.dim() == 2:
        T_dim, F_dim = augmented.shape
        for _ in range(num_freq_masks):
            f_len = torch.randint(0, min(freq_mask_param, F_dim), (1,), device=features.device).item()
            f0 = torch.randint(0, max(1, F_dim - f_len), (1,), device=features.device).item()
            augmented[:, f0:f0+f_len] = 0.0
        for _ in range(num_time_masks):
            t_len = torch.randint(0, min(time_mask_param, T_dim), (1,), device=features.device).item()
            t0 = torch.randint(0, max(1, T_dim - t_len), (1,), device=features.device).item()
            augmented[t0:t0+t_len, :] = 0.0
    elif augmented.dim() == 3:
        if augmented.shape[1] < augmented.shape[2]:
            B, F_dim, T_dim = augmented.shape
            for _ in range(num_freq_masks):
                f_len = torch.randint(0, min(freq_mask_param, F_dim), (1,), device=features.device).item()
                f0 = torch.randint(0, max(1, F_dim - f_len), (1,), device=features.device).item()
                augmented[:, f0:f0+f_len, :] = 0.0
            for _ in range(num_time_masks):
                t_len = torch.randint(0, min(time_mask_param, T_dim), (1,), device=features.device).item()
                t0 = torch.randint(0, max(1, T_dim - t_len), (1,), device=features.device).item()
                augmented[:, :, t0:t0+t_len] = 0.0
        else:
            B, T_dim, F_dim = augmented.shape
            for _ in range(num_freq_masks):
                f_len = torch.randint(0, min(freq_mask_param, F_dim), (1,), device=features.device).item()
                f0 = torch.randint(0, max(1, F_dim - f_len), (1,), device=features.device).item()
                augmented[:, :, f0:f0+f_len] = 0.0
            for _ in range(num_time_masks):
                t_len = torch.randint(0, min(time_mask_param, T_dim), (1,), device=features.device).item()
                t0 = torch.randint(0, max(1, T_dim - t_len), (1,), device=features.device).item()
                augmented[:, t0:t0+t_len, :] = 0.0
    return augmented


if __name__ == "__main__":
    print("🚀 Training Engine: Dual Distillation + 2-bit STE + SpecAugment")
