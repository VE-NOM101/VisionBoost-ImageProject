import cv2
import numpy as np
import json
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import os
def motion_blur_psf(length=15, angle=0):
    psf = np.zeros((length, length))
    center = length // 2

    # Draw a horizontal line first
    cv2.line(psf, (0, center), (length - 1, center), 1, thickness=1)

    # Rotate relative to horizontal
    rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1)
    psf = cv2.warpAffine(psf, rot_mat, (length, length))

    # Normalize
    psf = psf / psf.sum()
    return psf


def add_noise(image, noise_type="none", noise_level=0 ):

    noisy_image = image.copy().astype(np.float32)
    
    if noise_type.lower() == "gaussian":
        mean = 0
        sigma = (noise_level / 100) * 50  # adjust max sigma
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image += gauss

    elif noise_type.lower() == "salt_pepper":
        s_vs_p = 0.5 #salt vs pepper
        amount = (noise_level / 100) * 0.05  # max 5% pixels at noise_level=100
        # Salt
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 255
        # Pepper
        num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 0
    else:
        return image

    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


def apply_motion_blur(image_path, length=15, angle=0, noise_type="none", noise_level=0, output_dir="degraded"):
    # Read input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Generate PSF
    psf = motion_blur_psf(length, angle)

    # Apply motion blur
    blurred = cv2.filter2D(img, -1, psf)

    # Add noise
    degraded = add_noise(blurred, noise_type=noise_type, noise_level=noise_level)

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Unique ID using timestamp
    uid = str(int(time.time() * 1000))

    # Paths
    degraded_path = os.path.join(output_dir, f"degraded_{uid}.png")
    psf_json_path = os.path.join(output_dir, f"psf_{uid}.json")
    psf_img_path = os.path.join(output_dir, f"psf_{uid}.png")

    # Save degraded image
    cv2.imwrite(degraded_path, degraded)

    check_salt_pepper(degraded)
    # Save PSF kernel + noise info as JSON
    with open(psf_json_path, "w") as f:
        json.dump({
            "kernel": psf.tolist(),
            "length": length,
            "angle": angle,
            "noise_type": noise_type,
            "noise_level": noise_level
        }, f, indent=4)

    # Save PSF as an image (normalized for visibility)
    psf_vis = (psf / psf.max() * 255).astype(np.uint8)
    cv2.imwrite(psf_img_path, psf_vis)

    print(f"Degraded image saved at {degraded_path}")
    print(f"PSF JSON saved at {psf_json_path}")
    print(f"PSF image saved at {psf_img_path}")

    return degraded_path, psf_json_path, psf_img_path


def check_salt_pepper(image):
    total_pixels = image.shape[0] * image.shape[1]
    white_pixels = np.sum(np.all(image == [255, 255, 255], axis=-1))
    black_pixels = np.sum(np.all(image == [0, 0, 0], axis=-1))
    print(f"White pixels: {white_pixels} ({white_pixels/total_pixels*100:.2f}%)")
    print(f"Black pixels: {black_pixels} ({black_pixels/total_pixels*100:.2f}%)")


def apply_custom_H_motion_blur(image_path, a, b, snr_db=30, T=1.0, output_dir="custom_degraded"):
    # --- load image ---
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    M, N = img.shape[:2]

    # ============================================================
    # Create degradation function H(u,v) using a and b directly
    # ============================================================
    def make_H_motion(M, N, a, b, T):
        u = np.fft.fftfreq(M)
        v = np.fft.fftfreq(N)
        U, V = np.meshgrid(u, v, indexing='ij')
        
        # Direct use of a and b
        arg = (U * a + V * b)
        eps = 1e-12
        x = np.pi * arg
        
        # Sinc function
        sinc = np.sin(x) / np.where(np.abs(x) < eps, 1.0, x)
        sinc = np.where(np.abs(x) < eps, 1.0, sinc)
        
        # Motion blur transfer function
        H = T * sinc * np.exp(-1j * x)
        return H

    H = make_H_motion(M, N, a, b, T)

    # ============================================================
    # Apply degradation per color channel
    # ============================================================
    def apply_degradation_color(img, H, snr_db=None):
        degraded = np.zeros_like(img)
        for c in range(img.shape[2]):
            F = np.fft.fft2(img[..., c])
            G = H * F
            g = np.fft.ifft2(G).real
            g = np.clip(g, 0, 1)
            
            # Add noise if SNR is specified
            if snr_db:
                signal_power = np.mean(g**2)
                snr_linear = 10 ** (snr_db / 10)
                noise_power = signal_power / snr_linear
                noise = np.random.normal(0, np.sqrt(noise_power), g.shape)
                g = np.clip(g + noise, 0, 1)
            
            degraded[..., c] = g
        return degraded

    degraded = apply_degradation_color(img, H, snr_db=snr_db)

    # --- save outputs ---
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    blurred_path = os.path.join(output_dir, f"Hmotion_a{a}_b{b}_{base}.png")
    psf_img_path = os.path.join(output_dir, f"Hmotion_PSF_a{a}_b{b}_{base}.png")

    plt.imsave(blurred_path, np.clip(degraded, 0, 1))
    plt.imsave(psf_img_path, np.log1p(np.abs(np.fft.fftshift(H))), cmap="gray")

    return blurred_path, psf_img_path
