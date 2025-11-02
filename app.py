import streamlit as st
from PIL import Image
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
import json
import os
st.set_page_config(page_title="VisionBoost - Image Restoration", layout="wide")

# --- Title & Intro ---
st.title("VisionBoost - Image Restoration")
st.write("Non-Blind Deblurring and Denoising Project UI")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Upload & Degradation",
    "Non-Blind Deblurring",
    "Blind Deblurring",
    "Denoising",
    "Reduce Periodic Noise",
    "Histogram Enhancement",
    "Visualization & Comparison",
])

# --- Page: Upload & Degradation ---
if page == "Upload & Degradation":
    st.header("1. Upload & Degradation Simulation")

    # Tabs for two degradation modes
    tab1, tab2 = st.tabs(["🧩 Preset Motion Blur", "🧮 Custom H(u,v) Simulation"])

    # ============================================================
    # 🧩 TAB 1: Preset Motion Blur (Existing Logic)
    # ============================================================
    with tab1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="preset_upload")

        if uploaded_file is not None:
            img = Image.open(uploaded_file)

            # Resize for display
            max_width = 600
            w, h = img.size
            if w > max_width:
                h = int(h * max_width / w)
                w = max_width
            img_display = img.resize((w, h))

            st.image(img_display, caption="Original Image", width=600)

            st.subheader("Degradation Parameters")
            blur_size = st.slider("Motion Blur Length (pixels)", 3, 51, 15, step=2, key="preset_len")
            blur_angle = st.slider("Motion Blur Angle (degrees)", -90, 90, 0, key="preset_angle")
            noise_type = st.selectbox("Noise Type", ["None", "Gaussian", "Salt Pepper"], key="preset_noise")
            noise_level = st.slider("Noise Level (%)", 0, 100, 0, key="preset_level")

            if st.button("Apply Preset Degradation", key="preset_apply"):
                from upload_degradation import apply_motion_blur

                temp_path = "temp_uploaded_image.png"
                img.save(temp_path)

                blurred_path, psf_json_path, psf_img_path = apply_motion_blur(
                    temp_path,
                    length=blur_size,
                    angle=blur_angle,
                    noise_type=noise_type.lower().replace(" ", "_"),
                    noise_level=noise_level,
                    output_dir="degraded"
                )

                st.success(f"✅ Degraded image saved: {blurred_path}")
                st.success(f"✅ PSF saved: {psf_json_path} & {psf_img_path}")

                degraded_img = Image.open(blurred_path)
                w, h = degraded_img.size
                if w > max_width:
                    h = int(h * max_width / w)
                    w = max_width
                degraded_display = degraded_img.resize((w, h))
                st.image(degraded_display, caption="Degraded Image", width=600)

                psf_img = Image.open(psf_img_path)
                st.image(psf_img, caption="PSF (visual)", width=400)

    # ============================================================
    # 🧮 TAB 2: Custom H(u,v) Degradation (Mathematical Function)
    # ============================================================
    with tab2:
        uploaded_file_2 = st.file_uploader(
            "Upload an image", type=["jpg", "png", "jpeg"], key="custom_upload"
        )

        if uploaded_file_2 is not None:
            img = Image.open(uploaded_file_2)
            st.image(img, caption="Original Image", width=600)

            st.subheader("Custom H(u,v) Parameters")
            blur_size = st.slider("Motion Blur Length (pixels)", 1, 50, 15)
            blur_angle = st.slider("Motion Blur Angle (degrees, anticlockwise)", -90, 90, 0)
            snr_db = st.slider("SNR (dB)", 10, 50, 30)
            T = st.number_input("Exposure Time (T)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            if st.button("Apply H(u,v) Degradation"):
                from upload_degradation import apply_custom_H_motion_blur

                temp_path = "temp_uploaded_image.png"
                img.save(temp_path)

                blurred_path, psf_img_path = apply_custom_H_motion_blur(
                    temp_path,
                    length=blur_size,
                    angle=blur_angle,
                    snr_db=snr_db,
                    T=T,
                    output_dir="custom_degraded"
                )

                st.success(f"✅ Degraded image saved: {blurred_path}")
                st.success(f"✅ PSF visualization saved: {psf_img_path}")

                degraded_img = Image.open(blurred_path)
                st.image(degraded_img, caption="Degraded Image (via H(u,v))", width=600)

                psf_img = Image.open(psf_img_path)
                st.image(psf_img, caption="|H(u,v)| (log magnitude)", width=400)


# --- Page: Deblurring ---
elif page == "Non-Blind Deblurring":

    method = st.radio("Choose Deblurring Method", [
                      "Wiener Filtering","Wiener Filtering (Uniform Motion)", "Richardson-Lucy"])

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
    
    elif method == "Wiener Filtering (Uniform Motion)":
        st.subheader("Wiener Restoration – Uniform Linear Motion Model")

        uploaded_img = st.file_uploader(
            "Upload degraded / motion-blurred image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="wiener_uniform_img"
        )

        if uploaded_img is not None:
            img = io.imread(uploaded_img)
            img = img_as_float(img)
            st.image(img, caption="Blurred Input", use_container_width=True)

            st.markdown("### Motion Parameters")
            L = st.slider("Motion Blur Length (pixels)", 1, 50, 15)
            theta = st.slider("Motion Blur Angle (° anticlockwise)", -90, 90, 0)
            T = st.number_input("Exposure Time (T)", 0.1, 5.0, 1.0, step=0.1)
            snr_db = st.slider("Estimated SNR (dB)", 10, 50, 30)

            if st.button("🚀 Run Wiener Motion Deblurring"):
                from wiener_deconvolution import wiener_restore_uniform_motion

                restored, H = wiener_restore_uniform_motion(
                    img, L=L, theta=theta, T=T, snr_db=snr_db
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Blurred Input", use_container_width=True)
                with col2:
                    st.image(restored, caption="Wiener Restored Output", use_container_width=True)

                os.makedirs("wiener_motion", exist_ok=True)
                restored_name = f"wnr_motion_{uploaded_img.name}"
                save_path = os.path.join("wiener_motion", restored_name)
                io.imsave(save_path, img_as_ubyte(restored))
                st.success(f"✅ Restored image saved as `{save_path}`")

                # Optional: show PSF spectrum
                import matplotlib.pyplot as plt
                H_mag = np.log1p(np.abs(np.fft.fftshift(H)))
                st.image(H_mag, caption="|H(u,v)| (log magnitude)", width=400)

elif page == "Blind Deblurring":

    method = st.radio("Choose Deblurring Method", ["Blind-Richardson-Lucy"])
    if method == "Blind-Richardson-Lucy":
        st.subheader("Blind Richardson-Lucy Deconvolution")
        st.info("ℹ️ Blind deconvolution estimates BOTH the sharp image AND the blur kernel (PSF) simultaneously.")
    
        # Step 1: Upload blurred/degraded image
        uploaded_img = st.file_uploader(
            "Upload degraded / blurred image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="blind_rl_img"
        )

        if uploaded_img is not None:
            # Read image as float
            img = io.imread(uploaded_img)
            img = img_as_float(img)
        
            st.image(img, caption="Blurred Input Image", use_container_width=True)

            # Import blind RL functions
            try:
                from richardson_lucy_deconvolution import (
                    blind_richardson_lucy, 
                    create_gaussian_psf, 
                    normalize_psf
                )
            
                st.success("✓ Blind Richardson-Lucy module loaded successfully.")
            except Exception as e:
                st.error(f"Failed to import blind RL functions: {e}")
                st.stop()

            # Configuration section
            st.markdown("### Configuration")
        
            col1, col2 = st.columns(2)
        
            with col1:
                # PSF size input
                st.markdown("**PSF Size to Estimate**")
                psf_height = st.number_input(
                    "PSF Height", 
                    min_value=3, 
                    max_value=51, 
                    value=15, 
                    step=2,
                    help="Must be odd number. Larger size = more flexible but slower"
                )
                psf_width = st.number_input(
                    "PSF Width", 
                    min_value=3, 
                    max_value=51, 
                    value=15, 
                    step=2,
                    help="Must be odd number"
                )   
            
                # Ensure odd dimensions
                if psf_height % 2 == 0:
                    psf_height += 1
                if psf_width % 2 == 0:
                    psf_width += 1
                
                psf_size = (psf_height, psf_width)
                st.info(f"PSF size: {psf_size[0]} × {psf_size[1]}")
        
            with col2:
                # Iteration controls
                st.markdown("**Iteration Settings**")
                steps = st.slider(
                    "Number of alternating iterations",
                    min_value=5, 
                    max_value=100, 
                    value=25, 
                    step=5,
                    help="More iterations = better results but slower"
                )
            
                inner_steps = st.slider(
                    "Inner RL steps per update",
                    min_value=1, 
                    max_value=5, 
                    value=1, 
                    step=1,
                    help="Usually 1 is sufficient"
                )
        
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                clip_output = st.checkbox(
                    "Clip output to [0, 1]", 
                    value=True,
                    help="Prevents pixel values outside valid range"
                )
            
                psf_regularization = st.checkbox(
                    "Enable PSF regularization", 
                    value=True,
                    help="Applies smoothing to PSF for stability - recommended for noisy images"
                )
            
                # Optional: Upload initial PSF guess
                use_custom_psf = st.checkbox(
                    "Use custom initial PSF guess (optional)",
                    value=False,
                    help="If unchecked, will use Gaussian initialization"
                )
            
                uploaded_psf = None
                if use_custom_psf:
                    uploaded_psf = st.file_uploader(
                        "Upload initial PSF JSON (optional)", 
                        type=["json"],
                        key="blind_rl_psf"
                    )
                
                    if uploaded_psf is not None:
                        try:
                            from richardson_lucy_deconvolution import load_psf_from_json
                            psf_init, _ = load_psf_from_json(uploaded_psf)
                            st.success("✓ Custom PSF loaded successfully.")
                        except Exception as e:
                            st.error(f"Failed to load PSF JSON: {e}")
                        psf_init = None
                    else:
                        psf_init = None
                else:
                    psf_init = None

            # Run button
            if st.button("🚀 Run Blind Deconvolution", type="primary"):
                with st.spinner("Processing... This may take a few moments..."):
                    try:
                        # Initial guess for image
                        x0 = img.copy()
                    
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                        status_text.text("Initializing blind deconvolution...")
                        progress_bar.progress(10)
                    
                        # Run blind Richardson-Lucy
                        status_text.text(f"Running {steps} iterations of blind RL...")
                        progress_bar.progress(20)
                    
                        deblurred, estimated_psf = blind_richardson_lucy(
                            observation=img,
                            x_0=x0,
                            psf_init=psf_init,
                            psf_size=psf_size,
                            steps=steps,
                            inner_steps=inner_steps,
                            clip=clip_output,
                            psf_regularization=psf_regularization
                        )
                    
                        progress_bar.progress(90)
                        status_text.text("Finalizing results...")
                    
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📊 Results")
                    
                        # Show images side by side
                        col1, col2 = st.columns(2)
                    
                        with col1:
                            st.image(
                                img, 
                                caption="Blurred Input", 
                                use_container_width=True
                            )
                    
                        with col2:
                            st.image(
                                deblurred, 
                                caption=f"Deblurred Output (Blind RL, {steps} iterations)", 
                                use_container_width=True
                            )
                    
                        # Show estimated PSF
                        st.markdown("### 🔍 Estimated Point Spread Function (PSF)")
                    
                        col_psf1, col_psf2, col_psf3 = st.columns([1, 2, 1])
                    
                        with col_psf2:
                            # Normalize PSF for visualization
                            psf_vis = (estimated_psf - estimated_psf.min()) / (estimated_psf.max() - estimated_psf.min() + 1e-12)
                        
                            st.image(
                                psf_vis, 
                                caption=f"Estimated PSF ({psf_size[0]}×{psf_size[1]})",
                                use_container_width=True,
                                clamp=True
                            )
                        
                            # PSF statistics
                            st.markdown("**PSF Statistics:**")
                            st.write(f"- Shape: {estimated_psf.shape}")
                            st.write(f"- Sum: {np.sum(estimated_psf):.6f} (should be ≈1.0)")
                            st.write(f"- Max value: {np.max(estimated_psf):.6f}")
                            st.write(f"- Min value: {np.min(estimated_psf):.6f}")
                    
                        progress_bar.progress(100)
                        status_text.text("✓ Deconvolution complete!")
                    
                        # Save results
                        st.markdown("---")
                        st.markdown("### 💾 Save Results")
                    
                        os.makedirs("blind_richardson_lucy", exist_ok=True)
                    
                        # Save deblurred image
                        original_name = uploaded_img.name
                        name_without_ext = os.path.splitext(original_name)[0]
                        ext = os.path.splitext(original_name)[1]
                    
                        restored_name = f"blind_rl_{name_without_ext}{ext}"
                        save_path_img = os.path.join("blind_richardson_lucy", restored_name)
                    
                        # Convert to suitable type before saving
                        if np.issubdtype(deblurred.dtype, np.floating):
                            io.imsave(save_path_img, img_as_ubyte(deblurred))
                        else:
                            io.imsave(save_path_img, deblurred)
                    
                        st.success(f"✓ Deblurred image saved as `{save_path_img}`")
                    
                        # Save estimated PSF as numpy array and JSON
                        psf_npy_path = os.path.join(
                            "blind_richardson_lucy", 
                            f"estimated_psf_{name_without_ext}.npy"
                        )
                        np.save(psf_npy_path, estimated_psf)
                    
                        # Save PSF as JSON (matching your format)
                        psf_json_path = os.path.join(
                            "blind_richardson_lucy", 
                            f"estimated_psf_{name_without_ext}.json"
                        )
                    
                        psf_data = {
                            "type": "estimated_blind_rl",
                            "size": list(psf_size),
                            "kernel": estimated_psf.tolist(),
                            "parameters": {
                                "iterations": steps,
                                "inner_steps": inner_steps,
                                "regularization": psf_regularization
                            }
                        }
                    
                        with open(psf_json_path, 'w') as f:
                            json.dump(psf_data, f, indent=2)
                    
                        st.success(f"✓ Estimated PSF saved as:\n- `{psf_npy_path}`\n- `{psf_json_path}`")
                    
                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                    
                        with col_dl1:
                            # Download deblurred image
                            with open(save_path_img, "rb") as file:
                                st.download_button(
                                    label="📥 Download Deblurred Image",
                                    data=file,
                                    file_name=restored_name,
                                    mime="image/png"
                                )
                    
                        with col_dl2:
                            # Download PSF JSON
                            st.download_button(
                                label="📥 Download Estimated PSF (JSON)",
                                data=json.dumps(psf_data, indent=2),
                                file_name=f"estimated_psf_{name_without_ext}.json",
                                mime="application/json"
                            )
                    
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                    
                    except Exception as e:
                        st.error(f"❌ Error during blind deconvolution: {e}")
                        st.exception(e)



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
elif page == "Reduce Periodic Noise":
    tab1, tab2 = st.tabs(["Periodic Noise Generation", "Periodic Noise Removal Demo"])

    with tab1:
        from periodic_noise import add_periodic_noise_freq

        uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("L") 
            img_np = np.array(img)
            st.image(img_np, caption="Original Image", use_container_width=False)
        
        # User inputs for frequency-domain noise
            offsets_text = st.text_input("Enter spike offsets as u,v pairs (comma-separated, e.g. 50,30;70,-40)", 
                                     "50,30;70,-40")
            percent_amp = st.slider("Amplitude (% of max FFT)", 1, 20, 5)
            save_name = st.text_input("Save noisy image as", "noisy_periodic.png")
        
            if st.button("Generate Periodic Noise"):
                # Parse offsets input
                try:
                    offsets = []
                    for pair in offsets_text.split(";"):
                        u,v = map(int, pair.split(","))
                        offsets.append((u,v))
                
                    noisy_img, mag_spectrum, path = add_periodic_noise_freq(
                        img_np, offsets, percent_amp=percent_amp/100, save_name=save_name
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(noisy_img.astype(np.uint8), caption="Noisy Image", use_container_width=True)
                    with col2:
                        st.image(mag_spectrum, caption="Magnitude Spectrum of the noisy image", use_container_width=True)

                    st.success(f"Noisy image saved at: {path}")
            
                except Exception as e:
                    st.error(f"Error parsing offsets or generating noise: {e}")
    with tab2:
        st.success('Hellow')

# --- Page: Segmentation & Edges ---
elif page == "Histogram Enhancement":
    edge_method = st.radio("Edge Detector", ["Canny", "Sobel"])
    threshold = st.slider("Threshold", 0, 255, 128)
    st.button("Run Edge Detection")


elif page == "Visualization & Comparison":
    edge_method = st.radio("Edge Detector", ["Canny", "Sobel"])
    threshold = st.slider("Threshold", 0, 255, 128)
    st.button("Run Edge Detection")
