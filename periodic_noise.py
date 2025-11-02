import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os




def add_periodic_noise_freq(image, offsets, percent_amp=0.05, save_folder="periodic_noise", save_name="noisy_image.png"):
    rows, cols = image.shape
    F = fftshift(fft2(image))

    max_val = np.max(np.abs(F))
    amp = percent_amp * max_val

    for u_offset, v_offset in offsets:
        F[rows//2 + v_offset, cols//2 - u_offset] += amp
        F[rows//2 - v_offset, cols//2 + u_offset] += amp

    noisy_image = np.abs(ifft2(ifftshift(F)))

    magnitude_spectrum = np.log1p(np.abs(F))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, save_name)
    cv2.imwrite(save_path, noisy_image.astype(np.uint8))

    return noisy_image, magnitude_spectrum, save_path