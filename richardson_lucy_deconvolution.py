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