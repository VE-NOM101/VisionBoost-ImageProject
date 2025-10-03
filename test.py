import cv2
import numpy as np

def add_noise(image, noise_type="none", noise_level=10):
    """
    Add noise to an image.
    
    Parameters:
        image (numpy.ndarray): Input image (uint8).
        noise_type (str): "gaussian", "salt-pepper", or "none".
        noise_level (int): Noise intensity from 0 to 100.
        
    Returns:
        noisy_image (numpy.ndarray): Image with noise added.
    """
    noisy_image = image.copy().astype(np.float32)
    
    if noise_type.lower() == "gaussian":
        mean = 0
        sigma = (noise_level / 100) * 50  # adjust max sigma
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image += gauss

    elif noise_type.lower() == "salt-pepper":
        s_vs_p = 0.5
        amount = (noise_level / 100) * 0.05  # max 5% pixels at noise_level=100
        # Salt
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 255
        # Pepper
        num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 0

    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

# ---- Example usage ----
if __name__ == "__main__":
    img = cv2.imread("Lena.jpg")
    
    noisy_gaussian = add_noise(img, noise_type="gaussian", noise_level=20)
    noisy_sp = add_noise(img, noise_type="salt-pepper", noise_level=80)
    
    cv2.imwrite("noisy_gaussian.jpg", noisy_gaussian)
    cv2.imwrite("noisy_salt_pepper.jpg", noisy_sp)
