# --- place these imports near the top of your app ---
import os
import json
import numpy as np
from pathlib import Path
import streamlit as st
from skimage import io, img_as_float, img_as_ubyte
from scipy.signal import medfilt
from numpy.fft import fft2, ifft2
# --- helper functions ---
def load_psf_from_json(uploaded_psf):
    # uploaded_psf is a BytesIO object, so use read() then json.loads
    psf_bytes = uploaded_psf.read()           # read bytes from uploaded file
    psf_data = json.loads(psf_bytes.decode()) # decode bytes to string, then load JSON
    psf = np.array(psf_data["kernel"], dtype=np.float64)
    if psf.ndim != 2:
        raise ValueError("PSF kernel must be 2D.")
    return psf, psf_data   # no normalization, since already normalized


def wiener_deconv(blurred, psf, K=0.001):
    M, N = blurred.shape
    P, Q = 2*M, 2*N
    fp = np.zeros((P, Q), dtype=np.float32)
    fp[:M, :N] = blurred

    s, t = psf.shape
    pad_psf = np.zeros((P, Q), dtype=np.float32)
    pad_psf[:s, :t] = psf
    pad_psf = np.roll(pad_psf, -s//2, axis=0)
    pad_psf = np.roll(pad_psf, -t//2, axis=1)

    F = fft2(fp)
    H = fft2(pad_psf)

    H_conj = np.conj(H)
    denom = (H * H_conj) + K
    F_prime = (F * H_conj) / denom

    output = np.real(ifft2(F_prime))
    output = output[:M, :N]
    return output



def wiener_deconv_auto(blurred, psf, K=0.001):
    """
    Wiener deconvolution for grayscale or color images.
    
    blurred : 2D (H,W) or 3D (H,W,C) image
    psf     : 2D PSF
    K       : Wiener constant (NSR)
    
    Returns restored image with same shape and dtype as input.
    """
    # Save input dtype and scale
    orig_dtype = blurred.dtype
    min_val, max_val = blurred.min(), blurred.max()
    
    # Convert to float32 for processing
    blurred_f = blurred.astype(np.float32)
    
    # If grayscale
    if blurred.ndim == 2:
        restored_f = wiener_deconv(blurred_f, psf, K)
    # If color
    elif blurred.ndim == 3:
        restored_f = np.zeros_like(blurred_f)
        for c in range(blurred_f.shape[2]):
            restored_f[..., c] = wiener_deconv(blurred_f[..., c], psf, K)
    else:
        raise ValueError("Input image must be 2D or 3D array")
    
    # Clip to original range
    restored_f = np.clip(restored_f, min_val, max_val)
    
    # Convert back to original dtype
    if np.issubdtype(orig_dtype, np.integer):
        restored = restored_f.round().astype(orig_dtype)
    else:
        restored = restored_f.astype(orig_dtype)
    
    return restored
