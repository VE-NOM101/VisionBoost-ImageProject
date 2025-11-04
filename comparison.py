import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity as ssim


def mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE) between two images.
    Lower is better (0 = perfect match).

    Parameters:
    - img1: first image (numpy array)
    - img2: second image (numpy array)

    Returns:
    - mse_val: mean squared error value
    """
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


def psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.
    Higher is better (>30 dB is considered good).

    Parameters:
    - img1: first image (numpy array)
    - img2: second image (numpy array)

    Returns:
    - psnr_val: PSNR value in decibels (dB)
    """
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_val))


def ssim_score(img1, img2, multichannel=False):
    """
    Calculate Structural Similarity Index (SSIM).
    Ranges from -1 to 1, where 1 = perfect match.

    Parameters:
    - img1: first image (numpy array)
    - img2: second image (numpy array)
    - multichannel: True for color images, False for grayscale

    Returns:
    - ssim_val: SSIM value
    """
    if multichannel and len(img1.shape) == 3:
        return ssim(img1, img2, channel_axis=2, data_range=255)
    else:
        return ssim(img1, img2, data_range=255)


def compare_images(img1, img2):
    """
    Compare two images using MSE, PSNR, and SSIM metrics.
    Automatically detects if images are grayscale or color.

    Parameters:
    - img1: original image (numpy array)
    - img2: recovered/processed image (numpy array)

    Returns:
    - metrics: dictionary with 'mse', 'psnr', 'ssim', and 'quality_level'
    """
    # Ensure images have same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same dimensions. Got {img1.shape} and {img2.shape}")

    # Determine if color or grayscale
    is_color = len(img1.shape) == 3 and img1.shape[2] == 3

    # Calculate metrics
    mse_val = mse(img1, img2)
    psnr_val = psnr(img1, img2)
    ssim_val = ssim_score(img1, img2, multichannel=is_color)

    # Determine overall quality level
    if psnr_val == float('inf'):
        quality_level = "Perfect Match"
    elif psnr_val > 40:
        quality_level = "Excellent"
    elif psnr_val > 30:
        quality_level = "Good"
    elif psnr_val > 20:
        quality_level = "Fair"
    else:
        quality_level = "Poor"

    return {
        'mse': mse_val,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'quality_level': quality_level
    }


def get_metric_interpretation(metric_name, value):
    """
    Get color and interpretation for a metric value.

    Parameters:
    - metric_name: 'mse', 'psnr', or 'ssim'
    - value: metric value

    Returns:
    - color: 'green', 'orange', or 'red'
    - interpretation: 'Excellent', 'Good', 'Fair', or 'Poor'
    """
    if metric_name == 'mse':
        if value < 100:
            return 'green', 'Excellent'
        elif value < 500:
            return 'orange', 'Good'
        elif value < 1000:
            return 'orange', 'Fair'
        else:
            return 'red', 'Poor'

    elif metric_name == 'psnr':
        if value == float('inf'):
            return 'green', 'Perfect'
        elif value > 40:
            return 'green', 'Excellent'
        elif value > 30:
            return 'green', 'Good'
        elif value > 20:
            return 'orange', 'Fair'
        else:
            return 'red', 'Poor'

    elif metric_name == 'ssim':
        if value > 0.95:
            return 'green', 'Excellent'
        elif value > 0.85:
            return 'green', 'Good'
        elif value > 0.7:
            return 'orange', 'Fair'
        else:
            return 'red', 'Poor'

    return 'gray', 'Unknown'


def calculate_difference_map(img1, img2):
    """
    Calculate absolute difference between two images for visualization.

    Parameters:
    - img1: first image (numpy array)
    - img2: second image (numpy array)

    Returns:
    - diff_map: absolute difference image (enhanced for visibility)
    """
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))

    # Enhance for visibility (scale up small differences)
    diff_enhanced = np.clip(diff * 5, 0, 255).astype(np.uint8)

    return diff_enhanced