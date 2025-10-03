import cv2
import numpy as np

# --- 1. Gaussian Filter ---
def gaussian_filter(img, kernel_size=(3,3), sigma=1):
    return cv2.GaussianBlur(img, kernel_size, sigma)


# --- 2. Median Filter ---
def median_filter(img, kernel_size=3):
    if len(img.shape) == 2:  # grayscale
        return cv2.medianBlur(img, kernel_size)
    else:  # color
        channels = cv2.split(img)
        channels = [cv2.medianBlur(c, kernel_size) for c in channels]
        return cv2.merge(channels)


# --- 3. Bilateral Filter ---
def bilateral_filter(img, neighbor_size=5, sigma_s=75, sigma_c=75):
    return cv2.bilateralFilter(img, d=neighbor_size, sigmaColor=sigma_c, sigmaSpace=sigma_s)


# --- 4. Non-Local Means Filter ---
def nlm_filter(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    if len(img.shape) == 2:  # grayscale
        return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
    else:  # color
        return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)


