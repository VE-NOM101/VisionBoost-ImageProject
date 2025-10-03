import cv2
import numpy as np
import json
import os
import time

def motion_blur_psf(length=15, angle=0):
    """
    Generate a motion blur PSF (Point Spread Function).
    length: size of the motion
    angle: direction of the motion in degrees (relative to horizontal).
    """
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
    """
    Add noise to an image.
    noise_type: 'gaussian', 'salt_pepper', or 'none'
    noise_level: 0-100 (percentage scale)
                 - Gaussian: variance scales with noise_level
                 - Salt & Pepper: fraction of corrupted pixels = noise_level/100
    mean: mean for Gaussian noise
    """
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

