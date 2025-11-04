
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os


def add_periodic_noise_grayscale(img, offsets, percent_amp=0.05, save_name="noisy_periodic.png"):
    """
    Add periodic noise to grayscale image in frequency domain.

    Parameters:
    - img: grayscale image (numpy array)
    - offsets: list of (u, v) offset tuples for spike locations
    - percent_amp: amplitude as percentage of max FFT magnitude
    - save_name: output filename

    Returns:
    - noisy_img: noisy image
    - mag_spectrum: log magnitude spectrum (for visualization)
    - save_path: path where image was saved
    """
    rows, cols = img.shape
    F = fftshift(fft2(img))
    max_val = np.max(np.abs(F))
    amp = percent_amp * max_val

    # Add symmetric spikes at each offset
    for u_offset, v_offset in offsets:
        F[rows//2 + v_offset, cols//2 + u_offset] += amp
        F[rows//2 - v_offset, cols//2 - u_offset] += amp

    # Generate noisy image
    noisy_img = np.abs(ifft2(ifftshift(F)))

    # Clip values to [0, 255] to prevent overflow
    noisy_img = np.clip(noisy_img, 0, 255)

    # Compute magnitude spectrum for visualization
    mag_spectrum = np.log1p(np.abs(F))
    mag_spectrum = cv2.normalize(mag_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save image
    output_folder = "noisy_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    save_path = os.path.join(output_folder, save_name)
    cv2.imwrite(save_path, noisy_img.astype(np.uint8))

    return noisy_img, mag_spectrum, save_path


def detect_periodic_noise_spikes(F, rows, cols, dc_mask_radius=10, threshold_factor=3, nms_kernel_size=21):
    """
    Detect periodic noise spike locations in frequency domain.

    Parameters:
    - F: shifted FFT of image
    - rows, cols: image dimensions
    - dc_mask_radius: radius to mask DC component
    - threshold_factor: number of std deviations above mean for thresholding
    - nms_kernel_size: kernel size for non-maximum suppression

    Returns:
    - coords: list of (x, y) coordinates of detected spikes
    - magnitude_spectrum: log magnitude spectrum
    - binary_mask: thresholded binary mask
    - nms: result after non-maximum suppression
    """
    # Compute log magnitude spectrum
    magnitude_spectrum = np.log1p(np.abs(F))

    # Mask DC component
    mask = np.ones_like(magnitude_spectrum)
    cv2.circle(mask, (cols//2, rows//2), dc_mask_radius, 0, -1)
    spectrum_masked = magnitude_spectrum * mask

    # Adaptive threshold
    mean_val = np.mean(spectrum_masked)
    std_val = np.std(spectrum_masked)
    T = mean_val + threshold_factor * std_val
    binary_mask = (spectrum_masked > T).astype(np.uint8)

    # Non-Maximum Suppression
    norm = cv2.normalize(spectrum_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = np.ones((nms_kernel_size, nms_kernel_size), np.uint8)
    local_max = cv2.dilate(norm, kernel)
    nms = np.zeros_like(norm, np.uint8)
    nms[(norm == local_max) & (binary_mask == 1)] = 255

    # Extract coordinates
    coords = cv2.findNonZero(nms)
    if coords is not None:
        coords = [tuple(pt[0]) for pt in coords]
    else:
        coords = []

    return coords, magnitude_spectrum, binary_mask, nms


def butterworth_notch_reject(shape, D0=10, n=2, notch_centers=[]):
    """
    Create Butterworth notch reject filter.

    Parameters:
    - shape: (rows, cols) of image
    - D0: notch radius
    - n: filter order
    - notch_centers: list of (x, y) coordinates to reject

    Returns:
    - H: Butterworth notch filter
    """
    rows, cols = shape
    H = np.ones((rows, cols), dtype=np.float32)
    u = np.arange(rows) - rows/2
    v = np.arange(cols) - cols/2
    V, U = np.meshgrid(v, u)

    for (x, y) in notch_centers:
        u_k = y - cols/2
        v_k = x - rows/2
        Dk = np.sqrt((U - u_k)**2 + (V - v_k)**2)
        Dk_neg = np.sqrt((U + u_k)**2 + (V + v_k)**2)
        H *= 1 / (1 + ((D0**2) / (Dk * Dk_neg + 1e-6))**n)

    return H


def remove_periodic_noise_grayscale(noisy_image, D0=10, n=2, dc_mask_radius=10, 
                                     threshold_factor=3, nms_kernel_size=21):
    """
    Remove periodic noise from grayscale image using Butterworth notch reject filter.

    Parameters:
    - noisy_image: input noisy grayscale image
    - D0: notch radius for Butterworth filter
    - n: Butterworth filter order
    - dc_mask_radius: radius to mask DC component during spike detection
    - threshold_factor: threshold parameter for spike detection
    - nms_kernel_size: kernel size for non-maximum suppression

    Returns:
    - recovered_image: denoised image
    - coords: detected spike coordinates
    - H: Butterworth notch filter used
    - magnitude_spectrum: log magnitude spectrum
    - binary_mask: thresholded mask
    - nms: non-maximum suppression result
    """
    rows, cols = noisy_image.shape[:2]

    # Detect spikes
    F = fftshift(fft2(noisy_image))
    coords, magnitude_spectrum, binary_mask, nms = detect_periodic_noise_spikes(
        F, rows, cols, dc_mask_radius, threshold_factor, nms_kernel_size
    )

    if len(coords) == 0:
        return noisy_image, coords, np.ones((rows, cols)), magnitude_spectrum, binary_mask, nms

    # Create and apply Butterworth notch filter
    H = butterworth_notch_reject((rows, cols), D0, n, coords)
    F_filtered = F * H
    recovered_image = np.abs(ifft2(ifftshift(F_filtered)))

    # Clip values to [0, 255]
    recovered_image = np.clip(recovered_image, 0, 255)

    return recovered_image, coords, H, magnitude_spectrum, binary_mask, nms