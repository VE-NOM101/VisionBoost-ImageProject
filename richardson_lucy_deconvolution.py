import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

# --- helper functions ---
def load_psf_from_json(uploaded_psf):
    # uploaded_psf is a BytesIO object, so use read() then json.loads
    psf_bytes = uploaded_psf.read()           # read bytes from uploaded file
    psf_data = json.loads(psf_bytes.decode()) # decode bytes to string, then load JSON
    psf = np.array(psf_data["kernel"], dtype=np.float64)
    if psf.ndim != 2:
        raise ValueError("PSF kernel must be 2D.")
    return psf, psf_data   # no normalization, since already normalized

def richardson_lucy_np(observation, x_0, psf, steps=10, clip=True, filter_epsilon=1e-12):
    """
    Richardson-Lucy deconvolution using NumPy and OpenCV.

    Args:
        observation (np.ndarray): Observed blurred image (H x W) or (H x W x C) for RGB.
        x_0 (np.ndarray): Initial estimate of the deconvolved image.
        psf (np.ndarray): Point Spread Function (PSF) kernel.
        steps (int): Number of iterations.
        clip (bool): Whether to clip the output between 0 and 1.
        filter_epsilon (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Deconvolved image.
    """

    # Ensure float
    observation = observation.astype(np.float32)
    im_deconv = x_0.astype(np.float32)
    psf_flip = cv2.flip(psf, -1)  # flip both axes (like torch.flip dims=[2,3])

    # Padding sizes
    pad_y = psf.shape[0] // 2
    pad_x = psf.shape[1] // 2

    for _ in range(steps):
        # Convolve current estimate with PSF
        if im_deconv.ndim == 2:  # grayscale
            conv = cv2.filter2D(im_deconv, -1, psf, borderType=cv2.BORDER_REPLICATE) + 1e-12
        else:  # RGB
            conv = np.zeros_like(im_deconv)
            for c in range(im_deconv.shape[2]):
                conv[:, :, c] = cv2.filter2D(im_deconv[:, :, c], -1, psf, borderType=cv2.BORDER_REPLICATE)
            conv += 1e-12

        # Compute relative blur
        relative_blur = observation / np.maximum(conv, filter_epsilon)

        # Back-project with flipped PSF
        if im_deconv.ndim == 2:
            correction = cv2.filter2D(relative_blur, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
            im_deconv *= correction
        else:
            correction = np.zeros_like(im_deconv)
            for c in range(im_deconv.shape[2]):
                correction[:, :, c] = cv2.filter2D(relative_blur[:, :, c], -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
            im_deconv *= correction

    if clip:
        im_deconv = np.clip(im_deconv, 0, 1)

    return im_deconv


def blind_richardson_lucy(observation, x_0=None, psf_init=None, psf_size=(15, 15), 
                         steps=20, inner_steps=1, clip=True, filter_epsilon=1e-12,
                         psf_regularization=True):
    """
    Blind Richardson-Lucy deconvolution - estimates both image AND PSF.
    
    Args:
        observation: Observed blurred image (H x W) or (H x W x C)
        x_0: Initial image estimate (default: observation)
        psf_init: Initial PSF estimate (default: Gaussian)
        psf_size: Size of PSF to estimate, e.g., (15, 15)
        steps: Number of alternating iterations
        inner_steps: RL iterations per update (typically 1)
        clip: Whether to clip output [0, 1]
        filter_epsilon: Avoid division by zero
        psf_regularization: Apply PSF smoothing for stability
    
    Returns:
        tuple: (deconvolved_image, estimated_psf)
    """
    observation = observation.astype(np.float32)
    
    # Initialize image estimate
    if x_0 is None:
        im_estimate = observation.copy()
    else:
        im_estimate = x_0.astype(np.float32)
    
    # Initialize PSF estimate
    if psf_init is None:
        psf_estimate = create_gaussian_psf(psf_size)
    else:
        psf_estimate = normalize_psf(psf_init.astype(np.float32))
    
    is_color = observation.ndim == 3
    
    # Main alternating optimization loop
    for step in range(steps):
        # Step 1: Update image given current PSF
        for _ in range(inner_steps):
            im_estimate = _update_image(observation, im_estimate, psf_estimate, 
                                       filter_epsilon, is_color)
        
        # Step 2: Update PSF given current image
        for _ in range(inner_steps):
            psf_estimate = _update_psf(observation, im_estimate, psf_estimate, 
                                      filter_epsilon, is_color)
        
        # Normalize PSF (prevent drift)
        psf_estimate = normalize_psf(psf_estimate)
        
        # Regularization: smooth PSF periodically
        if psf_regularization and step % 5 == 0:
            psf_estimate = smooth_psf(psf_estimate, sigma=0.3)
            psf_estimate = normalize_psf(psf_estimate)
    
    if clip:
        im_estimate = np.clip(im_estimate, 0, 1)
    
    return im_estimate, psf_estimate


def _update_image(observation, im_estimate, psf_estimate, filter_epsilon, is_color):
    """Image update: f^(n+1) = f^(n) * [g(-) ⊗ (b / (g ⊗ f^(n)))]"""
    psf_flip = cv2.flip(psf_estimate, -1)
    
    if not is_color:
        conv = cv2.filter2D(im_estimate, -1, psf_estimate, borderType=cv2.BORDER_REPLICATE)
        conv = np.maximum(conv, filter_epsilon)
        relative_blur = observation / conv
        correction = cv2.filter2D(relative_blur, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
        im_estimate = im_estimate * correction
    else:
        for c in range(im_estimate.shape[2]):
            conv = cv2.filter2D(im_estimate[:, :, c], -1, psf_estimate, borderType=cv2.BORDER_REPLICATE)
            conv = np.maximum(conv, filter_epsilon)
            relative_blur = observation[:, :, c] / conv
            correction = cv2.filter2D(relative_blur, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
            im_estimate[:, :, c] = im_estimate[:, :, c] * correction
    
    return im_estimate


def _update_psf(observation, im_estimate, psf_estimate, filter_epsilon, is_color):
    """PSF update: g^(n+1) = g^(n) * [f(-) ⊗ (b / (f ⊗ g^(n)))]"""
    if not is_color:
        conv = cv2.filter2D(im_estimate, -1, psf_estimate, borderType=cv2.BORDER_REPLICATE)
        conv = np.maximum(conv, filter_epsilon)
        relative_blur = observation / conv
        correction = _correlate_for_psf(relative_blur, im_estimate, psf_estimate.shape)
        psf_estimate = psf_estimate * correction
    else:
        correction_sum = np.zeros_like(psf_estimate)
        for c in range(im_estimate.shape[2]):
            conv = cv2.filter2D(im_estimate[:, :, c], -1, psf_estimate, borderType=cv2.BORDER_REPLICATE)
            conv = np.maximum(conv, filter_epsilon)
            relative_blur = observation[:, :, c] / conv
            correction = _correlate_for_psf(relative_blur, im_estimate[:, :, c], psf_estimate.shape)
            correction_sum += correction
        psf_estimate = psf_estimate * (correction_sum / im_estimate.shape[2])
    
    return psf_estimate


def _correlate_for_psf(relative_blur, image, psf_shape):
    """Compute f(-) ⊗ relative_blur for PSF update"""
    h, w = psf_shape
    pad_h, pad_w = h // 2, w // 2
    
    relative_blur_pad = cv2.copyMakeBorder(relative_blur, pad_h, pad_h, pad_w, pad_w, 
                                          cv2.BORDER_REPLICATE)
    image_pad = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, 
                                  cv2.BORDER_REPLICATE)
    
    correlation = np.zeros(psf_shape, dtype=np.float32)
    img_h, img_w = relative_blur.shape
    
    for i in range(img_h):
        for j in range(img_w):
            patch = image_pad[i:i+h, j:j+w]
            patch_flip = cv2.flip(patch, -1)
            correlation += relative_blur_pad[i+pad_h, j+pad_w] * patch_flip
    
    correlation /= (img_h * img_w)
    return correlation


# Add these functions to richardson_lucy_deconvolution.py
def normalize_psf(psf):
    """Normalize PSF to sum to 1"""
    psf_sum = np.sum(psf)
    if psf_sum > 0:
        return psf / psf_sum
    return psf

def create_gaussian_psf(size, sigma=None):
    """Create initial Gaussian PSF estimate"""
    if sigma is None:
        sigma = min(size) / 6.0
    h, w = size
    center_h, center_w = h // 2, w // 2
    psf = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            psf[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
    return normalize_psf(psf)

def smooth_psf(psf, sigma=0.5):
    """Apply slight Gaussian smoothing to PSF for regularization"""
    kernel_size = 3
    psf_smooth = cv2.GaussianBlur(psf, (kernel_size, kernel_size), sigma)
    return psf_smooth