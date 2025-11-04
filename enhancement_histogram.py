
"""
Histogram Equalization Module
Author: Choyan Barua
Date: Oct 25, 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def calc_pdf(img):
    """
    Calculate Probability Density Function (PDF) from image histogram.

    Parameters:
    - img: grayscale image (numpy array)

    Returns:
    - pdf: probability density function
    """
    M, N = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    L, _ = hist.shape
    pdf = np.array([])
    for i in range(0, L):
        pdf = np.append(pdf, hist[i] / (M * N))

    return pdf


def calc_cdf(img):
    """
    Calculate Cumulative Distribution Function (CDF) from image PDF.

    Parameters:
    - img: grayscale image (numpy array)

    Returns:
    - cdf: cumulative distribution function
    """
    pdf = calc_pdf(img)
    cdf = np.array([])
    L = pdf.shape
    temp = 0
    for i in range(0, L[0]):
        temp += pdf[i]
        cdf = np.append(cdf, temp)

    return cdf


def histr_equalization(img):
    """
    Perform histogram equalization on grayscale image.

    Parameters:
    - img: grayscale image (numpy array)

    Returns:
    - equalized_img: histogram equalized image
    """
    cdf = calc_cdf(img)
    L = 256
    sk = np.zeros(L)
    for i in range(0, L):
        sk[i] = np.round((L - 1) * cdf[i])

    equalized_img = cv2.LUT(img, sk.astype(np.uint8))
    return equalized_img


def equalize_grayscale(img):
    """
    Equalize grayscale image and return visualization data.

    Parameters:
    - img: grayscale image (numpy array)

    Returns:
    - equalized_img: histogram equalized image
    - original_hist: histogram of original image
    - equalized_hist: histogram of equalized image
    - original_cdf: CDF of original image
    - equalized_cdf: CDF of equalized image
    """
    # Equalize
    equalized_img = histr_equalization(img)

    # Calculate histograms
    original_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

    # Calculate CDFs
    original_cdf = calc_cdf(img)
    equalized_cdf = calc_cdf(equalized_img)

    return equalized_img, original_hist, equalized_hist, original_cdf, equalized_cdf


def equalize_color_rgb(col_img):
    """
    Equalize color image by equalizing each RGB channel separately.

    Parameters:
    - col_img: color image in BGR format (numpy array)

    Returns:
    - col_img_eq: equalized color image
    - channel_data: dict with original and equalized channels and histograms
    """
    # Split channels
    b, g, r = cv2.split(col_img)

    # Equalize each channel
    b_eq = histr_equalization(b).astype(np.uint8)
    g_eq = histr_equalization(g).astype(np.uint8)
    r_eq = histr_equalization(r).astype(np.uint8)

    # Calculate histograms
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])

    b_eq_hist = cv2.calcHist([b_eq], [0], None, [256], [0, 256])
    g_eq_hist = cv2.calcHist([g_eq], [0], None, [256], [0, 256])
    r_eq_hist = cv2.calcHist([r_eq], [0], None, [256], [0, 256])

    # Merge equalized channels
    col_img_eq = cv2.merge([b_eq, g_eq, r_eq])

    # Prepare channel data for visualization
    channel_data = {
        'original_channels': {'b': b, 'g': g, 'r': r},
        'equalized_channels': {'b': b_eq, 'g': g_eq, 'r': r_eq},
        'original_hists': {'b': b_hist, 'g': g_hist, 'r': r_hist},
        'equalized_hists': {'b': b_eq_hist, 'g': g_eq_hist, 'r': r_eq_hist}
    }

    return col_img_eq, channel_data


def equalize_color_hsv(col_img):
    """
    Equalize color image by equalizing only the V (Value) channel in HSV space.
    This preserves color better than RGB equalization.

    Parameters:
    - col_img: color image in BGR format (numpy array)

    Returns:
    - hsv_eq_rgb: equalized color image in RGB format
    - v_channel: original V channel
    - v_eq: equalized V channel
    - v_cdf: CDF of original V channel
    - v_eq_cdf: CDF of equalized V channel
    """
    # Convert to HSV
    hsv = cv2.cvtColor(col_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Equalize only V channel
    v_eq = histr_equalization(v).astype(np.uint8)

    # Merge back
    hsv_eq = cv2.merge([h, s, v_eq])

    # Convert back to RGB for display
    hsv_eq_rgb = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    # Calculate CDFs
    v_cdf = calc_cdf(v)
    v_eq_cdf = calc_cdf(v_eq)

    return hsv_eq_rgb, v, v_eq, v_cdf, v_eq_cdf