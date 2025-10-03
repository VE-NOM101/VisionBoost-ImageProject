import streamlit as st
from PIL import Image
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
import json
import os
st.set_page_config(page_title="VisionBoost - Image Restoration", layout="wide")

# --- Title & Intro ---
st.title("VisionBoost 🔥 - Image Restoration")
st.write("Non-Blind Deblurring and Denoising Project UI")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Upload & Degradation",
    "Deblurring",
    "Denoising",
    "Segmentation & Edges",
    "Histogram Enhancement",
    "Frequency Domain",
    "Region Descriptors",
    "Quality Metrics",
    "Visualization & Comparison",
    "Color Image Restoration"
])

# --- Page: Upload & Degradation ---
# --- Page: Upload & Degradation ---
if page == "Upload & Degradation":
    st.header("1. Upload & Degradation Simulation")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        # Resize for medium display (max width 600px)
        max_width = 600
        w, h = img.size
        if w > max_width:
            h = int(h * max_width / w)
            w = max_width
        img_display = img.resize((w, h))

        st.image(img_display, caption="Original Image",  width=600)

        st.subheader("Degradation Parameters")
        blur_size = st.slider("Motion Blur Length (pixels)", 3, 51, 15, step=2)
        blur_angle = st.slider("Motion Blur Angle (degrees)", -90, 90, 0)
        noise_type = st.selectbox(
            "Noise Type", ["None", "Gaussian", "Salt Pepper"])
        noise_level = st.slider("Noise Level (%)", 0, 100, 0)

        if st.button("Apply Degradation"):
            from upload_degradation import apply_motion_blur

            # Save uploaded image temporarily
            temp_path = "temp_uploaded_image.png"
            img.save(temp_path)

            # Call function with noise options
            blurred_path, psf_json_path, psf_img_path = apply_motion_blur(
                temp_path,
                length=blur_size,
                angle=blur_angle,
                noise_type=noise_type.lower().replace(" ", "_"),
                noise_level=noise_level,
                output_dir="degraded"
            )

            st.success(f"Degraded image saved: {blurred_path}")
            st.success(f"PSF saved: {psf_json_path} & {psf_img_path}")

            # Display degraded image (resized medium)
            degraded_img = Image.open(blurred_path)
            w, h = degraded_img.size
            if w > max_width:
                h = int(h * max_width / w)
                w = max_width
            degraded_display = degraded_img.resize((w, h))
            st.image(degraded_display, caption="Degraded Image",  width=600)

            # Display PSF (small, fixed width)
            psf_img = Image.open(psf_img_path)
            st.image(psf_img, caption="PSF (visual)", width=400)


# --- Page: Deblurring ---
elif page == "Deblurring":
    st.header("2. Non-Blind Deblurring")

    method = st.radio("Choose Deblurring Method", [
                      "Wiener Filtering", "Richardson-Lucy"])

    if method == "Wiener Filtering":
        # Step 1: Upload blurred/degraded image
        uploaded_img = st.file_uploader(
            "Upload degraded / blurred image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
        )

        if uploaded_img is not None:
            # read image as float
            img = io.imread(uploaded_img)
            img = img_as_float(img)

            # Step 2: Upload PSF JSON
            uploaded_psf = st.file_uploader("Upload PSF JSON", type=["json"])

            try:
                from wiener_deconvolution import load_psf_from_json, wiener_deconv_auto

                psf, data = load_psf_from_json(uploaded_psf)

                print(psf)
            except Exception as e:
                st.error(f"Failed to load PSF JSON: {e}")

            nsr = st.number_input("Noise-to-Signal Ratio (NSR)",
                                  min_value=0.0, max_value=10.0, value=0.01, format="%.6f")
            st.write("Selected NSR:", nsr)

            if st.button("Run Deblurring"):
                deblurred = wiener_deconv_auto(img, psf, nsr)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Blurred Input",
                             use_container_width=True)

                with col2:
                    st.image(deblurred, caption="Deblurred Output",
                             use_container_width=True)

                os.makedirs("wiener", exist_ok=True)
                # Save restored image
                original_name = uploaded_img.name  # e.g., "myblur.png"
                restored_name = f"wnr_{original_name}"
                save_path = os.path.join("wiener", restored_name)

                # Convert to suitable type before saving
                if np.issubdtype(deblurred.dtype, np.floating):
                    # scale [0,1] -> uint8
                    io.imsave(save_path, img_as_ubyte(deblurred))
                else:
                    io.imsave(save_path, deblurred)

                st.success(f"Deblurred image saved as `{save_path}`")

    elif method == "Richardson-Lucy":
        # Step 1: Upload blurred/degraded image
        uploaded_img = st.file_uploader(
            "Upload degraded / blurred image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
        )

        if uploaded_img is not None:
            # read image as float
            img = io.imread(uploaded_img)
            img = img_as_float(img)

            # Step 2: Upload PSF JSON
            uploaded_psf = st.file_uploader("Upload PSF JSON", type=["json"])

            try:
                from richardson_lucy_deconvolution import load_psf_from_json, richardson_lucy_np

                psf, data = load_psf_from_json(uploaded_psf)
                st.success("PSF loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load PSF JSON: {e}")
                psf = None

            # Iteration control
            steps = st.slider("Number of RL iterations",
                              min_value=1, max_value=50, value=20, step=1)

            # Clip option (default = True)
            clip_output = st.checkbox("Clip output to [0,1]", value=True)

            if st.button("Run Deblurring") and psf is not None:
                # initial guess = blurred image
                x0 = img.copy()

                # run RL
                deblurred = richardson_lucy_np(
                    img, x0, psf, steps=steps, clip=clip_output)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Blurred Input",
                             use_container_width=True)

                with col2:
                    st.image(deblurred, caption="Deblurred Output (RL)",
                             use_container_width=True)

                # --- Save result ---
                os.makedirs("richardson_lucy", exist_ok=True)
                original_name = uploaded_img.name  # e.g., "myblur.png"
                restored_name = f"rl_{original_name}"
                save_path = os.path.join("richardson_lucy", restored_name)

                # Convert to suitable type before saving
                if np.issubdtype(deblurred.dtype, np.floating):
                    # scale [0,1] -> uint8
                    io.imsave(save_path, img_as_ubyte(deblurred))
                else:
                    io.imsave(save_path, deblurred)

                st.success(f"Deblurred image saved as `{save_path}`")


# --- Page: Denoising ---
elif page == "Denoising":
    st.header("3. Image Denoising")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
        st.image(img, caption="Uploaded Image", use_container_width=False)

        # Multi-select filters
        selected_filters = st.multiselect(
            "Choose Filters to Apply Sequentially",
            ["Mean", "Median", "Gaussian", "Bilateral", "Non-Local Means"]
        )

        # Dictionary to hold parameters for each filter
        filter_params = {}


        for f in selected_filters:
            if f == "Mean":
                st.subheader("Mean Filter Parameters")
                filter_params['Mean'] = {
                    'kernel_size': st.slider("Kernel Size", 1, 15, 3, step=2,key="mean_kernel")
                }

            elif f == "Median":
                st.subheader("Median Filter Parameters")
                filter_params['Median'] = {
                    'kernel_size': st.slider("Kernel Size", 1, 15, 3, step=2,key="median_kernel")
                }

            elif f == "Gaussian":
                st.subheader("Gaussian Filter Parameters")
                filter_params['Gaussian'] = {
                    'kernel_size': st.slider("Kernel Size", 1, 15, 3, step=2,key="gaussian_kernel"),
                    'sigma': st.number_input("Sigma", min_value=0.1, max_value=10.0, value=1.0, step=0.1,key="gaussian_sigma")
                }

            elif f == "Bilateral":
                st.subheader("Bilateral Filter Parameters")
                filter_params['Bilateral'] = {
                    'neighbor_size': st.slider("d (Neighborhood size)", 1, 20, 5, key="bilateral_d"),
                    'sigma_color': st.number_input("Sigma Color", min_value=1.0, max_value=150.0, value=75.0, step=1.0, key="bilateral_sigmaColor"),
                    'sigma_space': st.number_input("Sigma Space", min_value=1.0, max_value=150.0, value=75.0, step=1.0, key="bilateral_sigmaSpace")
                }

            elif f == "Non-Local Means":
                st.subheader("Non-Local Means Filter Parameters")
                filter_params['Non-Local Means'] = {
                    'h': st.slider("h (Filter Strength)", 1, 30, 10, key="nlm_h"),
                    'hColor': st.slider("hColor (Color Filter Strength)", 1, 30, 10,key="nlm_hColor"),
                    'templateWindowSize': st.slider("Template Window Size", 3, 15, 7, step=2, key="nlm_template"),
                    'searchWindowSize': st.slider("Search Window Size", 5, 35, 21, step=2, key="nlm_search")
                }


        # Apply filters sequentially
        from denoising import gaussian_filter,median_filter,nlm_filter,bilateral_filter

        if st.button("Run Denoising") and selected_filters:
            result = img_array.copy()
            for f in selected_filters:
                params = filter_params[f]
                if f == "Mean":
                    result = gaussian_filter(result, kernel_size=(
                        params['kernel_size'], params['kernel_size']))
                elif f == "Median":
                    result = median_filter(
                        result, kernel_size=params['kernel_size'])
                elif f == "Gaussian":
                    result = gaussian_filter(result, kernel_size=(
                        params['kernel_size'], params['kernel_size']), sigma=params['sigma'])
                elif f == "Bilateral":
                    result = bilateral_filter(result, neighbor_size=params['neighbor_size'],
                                          sigma_s=params['sigma_space'], sigma_c=params['sigma_color'])
                elif f == "Non-Local Means":
                    result = nlm_filter(result, h=params['h'], hColor=params['hColor'],
                                    templateWindowSize=params['templateWindowSize'],
                                    searchWindowSize=params['searchWindowSize'])
            st.image(result, caption="Denoised Image (Sequential Filters)",
                 use_container_width =True)
              # --- Save denoised image ---
            output_folder = "denoised"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Build output file name
            input_name = os.path.basename(uploaded_file.name)
            name, ext = os.path.splitext(input_name)
            output_path = os.path.join(output_folder, f"denoised_{name}.png")

            # Convert to BGR if color image for cv2.imwrite
            if len(result.shape) == 3 and result.shape[2] == 3:
                import cv2
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, result_bgr)
            else:
                import cv2
                cv2.imwrite(output_path, result)

            st.success(f"Denoised image saved to `{output_path}`")

# --- Page: Segmentation & Edges ---
elif page == "Segmentation & Edges":
    st.header("4. Segmentation & Edge Detection")
    edge_method = st.radio("Edge Detector", ["Canny", "Sobel"])
    threshold = st.slider("Threshold", 0, 255, 128)
    st.button("Run Edge Detection")

# --- Page: Histogram Enhancement ---
elif page == "Histogram Enhancement":
    st.header("5. Histogram-based Enhancement")
    option = st.selectbox(
        "Choose Method", ["Histogram Equalization", "Histogram Matching"])
    st.button("Apply Enhancement")

# --- Page: Frequency Domain ---
elif page == "Frequency Domain":
    st.header("6. Frequency Domain Processing")
    st.checkbox("Show Fourier Transform Spectrum")
    freq_filter = st.selectbox(
        "Frequency Filter", ["Low-pass", "High-pass", "Band-pass"])
    st.button("Apply Frequency Filter")

# --- Page: Region Descriptors ---
elif page == "Region Descriptors":
    st.header("7. Region Descriptors Extraction")
    st.write("Area, centroid, perimeter, bounding box etc. will be shown here.")
    st.button("Extract Descriptors")

# --- Page: Quality Metrics ---
elif page == "Quality Metrics":
    st.header("8. Quality Evaluation Metrics")
    st.write("Compare results using PSNR, SSIM")
    st.button("Evaluate Quality")

# --- Page: Visualization & Comparison ---
elif page == "Visualization & Comparison":
    st.header("9. Visualization & Comparison")
    st.write("Show Original vs Blurred vs Restored images side-by-side")
    st.button("Generate Comparison")

# --- Page: Color Image Restoration ---
elif page == "Color Image Restoration":
    st.header("10. Color Image Restoration")
    st.write("Process each RGB channel separately and recombine")
    st.button("Run Color Restoration")
