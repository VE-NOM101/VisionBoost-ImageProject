import streamlit as st
from PIL import Image
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
import json
import os
import cv2
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="VisionAID",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global Theme (no config file needed)
st.markdown("""
    <style>
        :root {
            --primary-color: #7b08ff;
            --secondary-color: #7b08ff;
            --text-color: #000000;
            --background-color: #ffffff;
            --secondary-background-color: #f7f3ff;
        }

        /* Buttons */
        div.stButton > button:first-child {
            background-color: var(--primary-color);
            color: white;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #5c00d9;
            color: white;
        }

        /* Sliders */
        .stSlider > div > div > div > div {
            background-color: var(--primary-color);
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color);
        }

        /* Radio / Checkbox active color */
        div[role='radiogroup'] label div[data-baseweb='radio']::before,
        div[data-testid='stCheckbox'] input:checked + div:before {
            border-color: var(--primary-color);
            background-color: var(--primary-color);
        }

        /* Sidebar header */
        section[data-testid="stSidebar"] > div:first-child {
            background-color: var(--secondary-background-color);
        }

        h1, h2, h3 {
            color: var(--primary-color);
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Upload & Degradation",
    "Non-Blind Deblurring",
    "Blind Deblurring",
    "Denoising",
    "Reduce Periodic Noise",
    "Histogram Enhancement",
    "Visualization & Comparison",
],index=0 )

if page == "Home":
    from home import render_home_page
    render_home_page()

# --- Page: Upload & Degradation ---
elif page == "Upload & Degradation":
    st.header("Upload & Degradation Simulation")

    # Tabs for two degradation modes
    tab1, tab2 = st.tabs(["Preset Motion Blur", "Linear Motion"])

    with tab1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="preset_upload")
        if uploaded_file is not None:
            img = Image.open(uploaded_file)

            # Resize for display
            max_width = 300  # smaller width for side-by-side
            w, h = img.size
            if w > max_width:
                h = int(h * max_width / w)
                w = max_width
                img_display = img.resize((w, h))

            # Create 2 columns: left for original image, right for parameters & degraded image
            col1, col2 = st.columns([1, 1])  # 50%-50% split

            with col1:
                st.image(img_display, caption="Original Image", width=w)

            with col2:
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

                st.success(f"Degraded image saved: {blurred_path}")
                st.success(f"PSF saved: {psf_json_path} & {psf_img_path}")


                degraded_img = Image.open(blurred_path)
                w2, h2 = degraded_img.size
                if w2 > max_width:
                    h2 = int(h2 * max_width / w2)
                    w2 = max_width
                    degraded_display = degraded_img.resize((w2, h2))

                psf_img = Image.open(psf_img_path)
                psf_display = psf_img.resize((300, 300))  # optional resize for PSF visualization

                #    Display degraded image and PSF side by side
                col3, col4 = st.columns([1, 1])
                with col3:
                    st.image(degraded_display, caption="Degraded Image", width=w2)
                with col4:
                    st.image(psf_display, caption="PSF (visual)", width=300)

    # ============================================================
    # 🧮 TAB 2: Custom H(u,v) Degradation (Mathematical Function)
    # ============================================================
    with tab2:
        uploaded_file_2 = st.file_uploader(
            "Upload an image", type=["jpg", "png", "jpeg"], key="custom_upload"
        )

        if uploaded_file_2 is not None:
            img = Image.open(uploaded_file_2)

            # Resize for display
            max_width = 300
            w, h = img.size
            if w > max_width:
                h = int(h * max_width / w)
                w = max_width
            img_display = img.resize((w, h))

            # --- Side by side: Original Image | Parameters ---
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img_display, caption="Original Image", width=w)
            with col2:
                st.subheader("Custom H(u,v) Parameters")
                blur_size = st.slider("Motion Blur Length (pixels)", 1, 50, 15)
                blur_angle = st.slider("Motion Blur Angle (degrees, anticlockwise)", -90, 90, 0)
                snr_db = st.slider("SNR (dB)", 10, 50, 30)
                T = st.number_input("Exposure Time (T)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

                apply_button = st.button("Apply H(u,v) Degradation")

            # --- Apply degradation ---
            if apply_button:
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

                st.success(f"Degraded image saved: {blurred_path}")
                st.success(f"PSF visualization saved: {psf_img_path}")

            # Load images and resize for side by side display
                degraded_img = Image.open(blurred_path)
                w2, h2 = degraded_img.size
                if w2 > max_width:
                    h2 = int(h2 * max_width / w2)
                    w2 = max_width
                degraded_display = degraded_img.resize((w2, h2))

                psf_img = Image.open(psf_img_path)
                psf_display = psf_img.resize((300, 300))  # optional fixed size

                # --- Side by side: Degraded Image | PSF ---
                col3, col4 = st.columns([1, 1])
                with col3:
                    st.image(degraded_display, caption="Degraded Image (via H(u,v))", width=w2)
                with col4:
                    st.image(psf_display, caption="|H(u,v)| (log magnitude)", width=300)


# --- Page: Deblurring ---
elif page == "Non-Blind Deblurring":

    st.header("Non-Blind Deblurring")

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
    st.header("Blind Deblurring")
    method = st.radio("Choose Deblurring Method", ["Blind-Richardson-Lucy"])
    if method == "Blind-Richardson-Lucy":
        st.subheader("Blind Richardson-Lucy Deconvolution")
        st.info("Blind deconvolution estimates BOTH the sharp image AND the blur kernel (PSF) simultaneously.")
    
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

# --- Page: Reduce Periodic Noise ---
elif page == "Reduce Periodic Noise":
    st.header("Periodic Noise Reduction (Grayscale)")
    tab1, tab2 = st.tabs(["Generate Periodic Noise", "Remove Periodic Noise"])

    # ========================================================================
    # TAB 1: Add Periodic Noise
    # ========================================================================
    with tab1:
        st.subheader("Add Periodic Noise to Grayscale Image")
        from periodic_noise import add_periodic_noise_grayscale

        uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "png", "jpeg"], key="noise_gen_upload")

        if uploaded_file is not None:
            # Load and convert to grayscale
            img_pil = Image.open(uploaded_file).convert('L')
            img_np = np.array(img_pil)

            st.image(img_np, caption="Original Grayscale Image", use_container_width=False, clamp=True)

            # Noise parameters
            st.markdown("**Noise Parameters**")
            offsets_text = st.text_input(
                "Enter spike offsets as u,v pairs (semicolon-separated)",
                "50,30;70,-40",
                help="Example: 50,30;70,-40 adds two noise sources"
            )
            percent_amp = st.slider("Amplitude (% of max FFT magnitude)", 1, 20, 5)
            save_name = st.text_input("Save noisy image as", "noisy_periodic.png")

            if st.button("Generate Periodic Noise"):
                try:
                    # Parse offsets
                    offsets = []
                    for pair in offsets_text.strip().split(";"):
                        u, v = map(int, pair.strip().split(","))
                        offsets.append((u, v))

                    # Generate noise
                    noisy_img, mag_spectrum, path = add_periodic_noise_grayscale(
                        img_np, offsets, percent_amp=percent_amp/100.0, save_name=save_name
                    )

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(noisy_img.astype(np.uint8), caption="Noisy Image", 
                               use_container_width=True, clamp=True)
                    with col2:
                        st.image(mag_spectrum, caption="FFT Magnitude Spectrum", 
                               use_container_width=True, clamp=True)

                    st.success(f"✓ Noisy image saved at: `{path}`")
                    st.info(f"Added {len(offsets)} symmetric noise spike(s) at offsets: {offsets}")

                except Exception as e:
                    st.error(f"Error generating periodic noise: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ========================================================================
    # TAB 2: Remove Periodic Noise
    # ========================================================================
    with tab2:
        st.subheader("Remove Periodic Noise using Butterworth Notch Filter")
        from periodic_noise import remove_periodic_noise_grayscale

        uploaded_noisy = st.file_uploader("Upload a noisy grayscale image", 
                                         type=["jpg", "png", "jpeg"], 
                                         key="noise_remove_upload")

        if uploaded_noisy is not None:
            # Load and convert to grayscale
            img_pil = Image.open(uploaded_noisy).convert('L')
            img_np = np.array(img_pil)

            st.image(img_np, caption="Uploaded Noisy Image", use_container_width=False, clamp=True)

            # Filter parameters
            st.markdown("**Filter Parameters**")
            col1, col2, col3 = st.columns(3)

            with col1:
                D0 = st.slider("Notch Radius (D0)", 1, 50, 10, 
                             help="Radius of the notch filter around detected spikes")
                n = st.slider("Filter Order (n)", 1, 10, 2, 
                            help="Higher order = sharper transition")

            with col2:
                dc_mask_radius = st.slider("DC Mask Radius", 5, 30, 10, 
                                          help="Radius to mask DC component during detection")
                threshold_factor = st.slider("Threshold Factor", 1.0, 10.0, 3.0, 0.5, 
                                            help="Sensitivity for spike detection (lower = more sensitive)")

            with col3:
                nms_kernel_size = st.slider("NMS Kernel Size", 5, 51, 21, 2, 
                                           help="Non-maximum suppression window size")

            # Output filename
            save_recovered_name = st.text_input("Save recovered image as", "recovered_image.png")

            if st.button("Remove Periodic Noise"):
                with st.spinner("Processing..."):
                    try:
                        # Process image
                        recovered, coords, H, mag_spec, binary_mask, nms_result = remove_periodic_noise_grayscale(
                            img_np,
                            D0=D0,
                            n=n,
                            dc_mask_radius=dc_mask_radius,
                            threshold_factor=threshold_factor,
                            nms_kernel_size=nms_kernel_size
                        )

                        # Save recovered image
                        output_folder = "recovered_images"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        save_path = os.path.join(output_folder, save_recovered_name)
                        cv2.imwrite(save_path, recovered.astype(np.uint8))

                        coords_list = [tuple(map(int, c)) for c in coords]
                        # Display results
                        st.success(f"✓ Detected {len(coords_list)} spike(s) at coordinates: {coords_list}")
                        st.success(f"✓ Recovered image saved at: `{save_path}`")

                        # Show comparison
                        st.markdown("### Results Comparison")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Noisy Image**")
                            st.image(img_np, use_container_width=True, clamp=True)

                        with col2:
                            st.markdown("**Recovered Image**")
                            st.image(recovered.astype(np.uint8), use_container_width=True, clamp=True)

                        # Show processing details
                        with st.expander("View Processing Details"):
                            st.markdown("### Frequency Domain Analysis")

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.markdown("**FFT Magnitude**")
                                mag_display = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                st.image(mag_display, use_container_width=True, clamp=True)

                            with col2:
                                st.markdown("**Thresholded Peaks**")
                                st.image(binary_mask*255, use_container_width=True, clamp=True)

                            with col3:
                                st.markdown("**After NMS**")
                                st.image(nms_result, use_container_width=True, clamp=True)

                            with col4:
                                st.markdown("**Butterworth Filter**")
                                H_display = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                st.image(H_display, use_container_width=True, clamp=True)

                    except Exception as e:
                        st.error(f"Error during noise removal: {e}")
                        import traceback
                        st.code(traceback.format_exc())
# --- Page: Segmentation & Edges ---
elif page == "Histogram Enhancement":
    st.header("Histogram Equalization")

    # Sub-tabs for different types of equalization
    tab1, tab2, tab3 = st.tabs(["Grayscale Equalization", "RGB Equalization", "HSV Equalization"])

    # ========================================================================
    # TAB 1: Grayscale Histogram Equalization
    # ========================================================================
    with tab1:
        st.subheader("Grayscale Histogram Equalization")
        from enhancement_histogram import equalize_grayscale

        uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "png", "jpeg"], key="gray_eq_upload")

        if uploaded_file is not None:
            # Load and convert to grayscale
            img_pil = Image.open(uploaded_file).convert('L')
            img_np = np.array(img_pil)

            st.image(img_np, caption="Original Grayscale Image", use_container_width=False, clamp=True)

            if st.button("Equalize Histogram", type="primary", key="gray_eq_btn"):
                with st.spinner("Processing..."):
                    try:
                        # Equalize
                        equalized_img, orig_hist, eq_hist, orig_cdf, eq_cdf = equalize_grayscale(img_np)

                        # Display comparison
                        st.markdown("### 📊 Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Image**")
                            st.image(img_np, use_container_width=True, clamp=True)

                        with col2:
                            st.markdown("**Equalized Image** ✨")
                            st.image(equalized_img, use_container_width=True, clamp=True)

                        # Histogram comparison
                        st.markdown("### 📈 Histogram Analysis")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Histogram**")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(orig_hist, color='red', linewidth=2)
                            ax.set_xlabel('Pixel Intensity')
                            ax.set_ylabel('Frequency')
                            ax.set_xlim([0, 256])
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close()

                        with col2:
                            st.markdown("**Equalized Histogram**")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(eq_hist, color='green', linewidth=2)
                            ax.set_xlabel('Pixel Intensity')
                            ax.set_ylabel('Frequency')
                            ax.set_xlim([0, 256])
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close()

                        # CDF comparison
                        with st.expander("📊 View CDF (Cumulative Distribution Function)", expanded=False):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Original CDF**")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.plot(orig_cdf, color='blue', linewidth=2)
                                ax.set_xlabel('Pixel Intensity')
                                ax.set_ylabel('CDF')
                                ax.set_xlim([0, 256])
                                ax.set_ylim([0, 1])
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close()

                            with col2:
                                st.markdown("**Equalized CDF**")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.plot(eq_cdf, color='orange', linewidth=2)
                                ax.set_xlabel('Pixel Intensity')
                                ax.set_ylabel('CDF')
                                ax.set_xlim([0, 256])
                                ax.set_ylim([0, 1])
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close()

                        # Save option
                        output_folder = "enhanced_images"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        save_path = os.path.join(output_folder, "gray_equalized.png")
                        cv2.imwrite(save_path, equalized_img)
                        st.success(f"✓ Equalized image saved at: `{save_path}`")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    # ========================================================================
    # TAB 2: RGB Color Histogram Equalization
    # ========================================================================
    with tab2:
        st.subheader("RGB Histogram Equalization")
        st.info("🎨 This method equalizes each RGB channel independently.")
        from enhancement_histogram import equalize_color_rgb

        uploaded_file = st.file_uploader("Upload a color image", type=["jpg", "png", "jpeg"], key="rgb_eq_upload")

        if uploaded_file is not None:
            # Load color image
            img_pil = Image.open(uploaded_file).convert('RGB')
            img_rgb = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            st.image(img_rgb, caption="Original Color Image", use_container_width=False, clamp=True)

            if st.button("Equalize RGB Channels", type="primary", key="rgb_eq_btn"):
                with st.spinner("Processing..."):
                    try:
                        # Equalize
                        col_img_eq, channel_data = equalize_color_rgb(img_bgr)
                        col_img_eq_rgb = cv2.cvtColor(col_img_eq, cv2.COLOR_BGR2RGB)

                        # Display comparison
                        st.markdown("### 📊 Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Image**")
                            st.image(img_rgb, use_container_width=True, clamp=True)

                        with col2:
                            st.markdown("**Equalized Image** ✨")
                            st.image(col_img_eq_rgb, use_container_width=True, clamp=True)

                        # Individual channel comparison
                        with st.expander("🔍 View Individual Channel Processing", expanded=True):
                            st.markdown("### Original vs Equalized Channels")

                            # Original channels
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Blue (Original)**")
                                st.image(channel_data['original_channels']['b'], use_container_width=True, clamp=True)
                            with col2:
                                st.markdown("**Green (Original)**")
                                st.image(channel_data['original_channels']['g'], use_container_width=True, clamp=True)
                            with col3:
                                st.markdown("**Red (Original)**")
                                st.image(channel_data['original_channels']['r'], use_container_width=True, clamp=True)

                            # Equalized channels
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Blue (Equalized)**")
                                st.image(channel_data['equalized_channels']['b'], use_container_width=True, clamp=True)
                            with col2:
                                st.markdown("**Green (Equalized)**")
                                st.image(channel_data['equalized_channels']['g'], use_container_width=True, clamp=True)
                            with col3:
                                st.markdown("**Red (Equalized)**")
                                st.image(channel_data['equalized_channels']['r'], use_container_width=True, clamp=True)

                            # Histogram comparison for Blue channel
                            st.markdown("### 📈 Blue Channel Histogram")
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(channel_data['original_hists']['b'], color='blue', 
                                   linewidth=2, label='Before Equalization', alpha=0.7)
                            ax.plot(channel_data['equalized_hists']['b'], color='orange', 
                                   linewidth=2, label='After Equalization', alpha=0.7)
                            ax.set_xlabel('Pixel Intensity')
                            ax.set_ylabel('Frequency')
                            ax.set_xlim([0, 256])
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close()

                        # Save option
                        output_folder = "enhanced_images"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        save_path = os.path.join(output_folder, "rgb_equalized.png")
                        cv2.imwrite(save_path, col_img_eq)
                        st.success(f"✓ Equalized image saved at: `{save_path}`")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    # ========================================================================
    # TAB 3: HSV Histogram Equalization (Better color preservation)
    # ========================================================================
    with tab3:
        st.subheader("HSV Histogram Equalization")
        st.info("🌈 This method equalizes only the V (Value/Brightness) channel, preserving color better.")
        from enhancement_histogram import equalize_color_hsv

        uploaded_file = st.file_uploader("Upload a color image", type=["jpg", "png", "jpeg"], key="hsv_eq_upload")

        if uploaded_file is not None:
            # Load color image
            img_pil = Image.open(uploaded_file).convert('RGB')
            img_rgb = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            st.image(img_rgb, caption="Original Color Image", use_container_width=False, clamp=True)

            if st.button("Equalize V Channel (HSV)", type="primary", key="hsv_eq_btn"):
                with st.spinner("Processing..."):
                    try:
                        # Equalize
                        hsv_eq_rgb, v_orig, v_eq, v_cdf, v_eq_cdf = equalize_color_hsv(img_bgr)

                        # Display comparison
                        st.markdown("### 📊 Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Image**")
                            st.image(img_rgb, use_container_width=True, clamp=True)

                        with col2:
                            st.markdown("**HSV Equalized Image** ✨")
                            st.image(hsv_eq_rgb, use_container_width=True, clamp=True)

                        st.success("✓ HSV equalization preserves hue and saturation, only adjusting brightness!")

                        # V channel comparison
                        with st.expander("🔍 View V (Value) Channel Processing", expanded=True):
                            st.markdown("### Original vs Equalized V Channel")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**V Channel (Original)**")
                                st.image(v_orig, use_container_width=True, clamp=True)
                            with col2:
                                st.markdown("**V Channel (Equalized)**")
                                st.image(v_eq, use_container_width=True, clamp=True)

                            # CDF comparison
                            st.markdown("### 📈 V Channel CDF")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(v_cdf, color='red', linewidth=2, label='Original CDF')
                            ax.plot(v_eq_cdf, color='green', linewidth=2, label='Equalized CDF')
                            ax.set_xlabel('Pixel Intensity')
                            ax.set_ylabel('CDF')
                            ax.set_xlim([0, 256])
                            ax.set_ylim([0, 1])
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close()

                        # Save option
                        output_folder = "enhanced_images"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        save_path = os.path.join(output_folder, "hsv_equalized.png")
                        # Convert back to BGR for saving
                        hsv_eq_bgr = cv2.cvtColor(hsv_eq_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, hsv_eq_bgr)
                        st.success(f"✓ Equalized image saved at: `{save_path}`")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())


elif page == "Visualization & Comparison":
    st.header("Image Quality Comparison")
    st.info("📊 Compare original vs recovered/processed images using MSE, PSNR, and SSIM metrics")

    from comparison import compare_images, get_metric_interpretation, calculate_difference_map

    # Two methods: Upload two images OR select from folders
    comparison_method = st.radio(
        "Select Comparison Method",
        ["Upload Two Images", "Select from Saved Folders"],
        horizontal=True
    )

    img1 = None
    img2 = None
    img1_name = "Image 1"
    img2_name = "Image 2"

    # ========================================================================
    # Method 1: Upload Two Images
    # ========================================================================
    if comparison_method == "Upload Two Images":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            uploaded_orig = st.file_uploader("Upload original image", 
                                            type=["jpg", "png", "jpeg"], 
                                            key="orig_upload")
            if uploaded_orig is not None:
                img_pil = Image.open(uploaded_orig)
                # Convert to numpy array
                if img_pil.mode == 'L':
                    img1 = np.array(img_pil)
                else:
                    img1 = np.array(img_pil.convert('RGB'))
                img1_name = "Original"
                st.image(img1, caption="Original Image", use_container_width=True, clamp=True)

        with col2:
            st.subheader("Recovered/Processed Image")
            uploaded_recovered = st.file_uploader("Upload recovered/processed image", 
                                                  type=["jpg", "png", "jpeg"], 
                                                  key="recovered_upload")
            if uploaded_recovered is not None:
                img_pil = Image.open(uploaded_recovered)
                # Convert to numpy array
                if img_pil.mode == 'L':
                    img2 = np.array(img_pil)
                else:
                    img2 = np.array(img_pil.convert('RGB'))
                img2_name = "Recovered"
                st.image(img2, caption="Recovered Image", use_container_width=True, clamp=True)

    # ========================================================================
    # Method 2: Select from Folders
    # ========================================================================
    else:
        import glob

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Select Original Image")
            # Look for images in common folders
            orig_folders = [".", "images", "original_images", "uploads"]
            orig_images = []
            for folder in orig_folders:
                if os.path.exists(folder):
                    orig_images.extend(glob.glob(os.path.join(folder, "*.jpg")))
                    orig_images.extend(glob.glob(os.path.join(folder, "*.png")))
                    orig_images.extend(glob.glob(os.path.join(folder, "*.jpeg")))

            if orig_images:
                selected_orig = st.selectbox("Choose original image", orig_images, key="select_orig")
                if selected_orig:
                    img1 = cv2.imread(selected_orig)
                    if img1 is not None:
                        if len(img1.shape) == 2:
                            pass  # Grayscale
                        else:
                            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                        img1_name = os.path.basename(selected_orig)
                        st.image(img1, caption=f"Original: {img1_name}", use_container_width=True, clamp=True)
            else:
                st.warning("No images found in common folders")

        with col2:
            st.subheader("Select Recovered Image")
            # Look for images in recovered/processed folders
            recovered_folders = ["recovered_images", "enhanced_images", "noisy_images", "denoised"]
            recovered_images = []
            for folder in recovered_folders:
                if os.path.exists(folder):
                    recovered_images.extend(glob.glob(os.path.join(folder, "*.jpg")))
                    recovered_images.extend(glob.glob(os.path.join(folder, "*.png")))
                    recovered_images.extend(glob.glob(os.path.join(folder, "*.jpeg")))

            if recovered_images:
                selected_recovered = st.selectbox("Choose recovered image", recovered_images, key="select_recovered")
                if selected_recovered:
                    img2 = cv2.imread(selected_recovered)
                    if img2 is not None:
                        if len(img2.shape) == 2:
                            pass  # Grayscale
                        else:
                            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        img2_name = os.path.basename(selected_recovered)
                        st.image(img2, caption=f"Recovered: {img2_name}", use_container_width=True, clamp=True)
            else:
                st.warning("No images found in recovered folders")

    # ========================================================================
    # Compare Images if Both are Loaded
    # ========================================================================
    if img1 is not None and img2 is not None:
        st.markdown("---")

        if st.button("🔍 Compare Images", type="primary"):
            with st.spinner("Calculating quality metrics..."):
                try:
                    # Ensure same dimensions
                    if img1.shape != img2.shape:
                        st.error(f"❌ Images must have same dimensions! "
                                f"Original: {img1.shape}, Recovered: {img2.shape}")
                    else:
                        # Calculate metrics
                        metrics = compare_images(img1, img2)

                        # Display overall quality
                        st.markdown("## 🎯 Overall Quality Assessment")
                        quality = metrics['quality_level']

                        if quality == "Perfect Match":
                            st.success(f"✨ **{quality}** - Images are identical!")
                        elif quality in ["Excellent", "Good"]:
                            st.success(f"✅ **{quality}** - High quality recovery!")
                        elif quality == "Fair":
                            st.warning(f"⚠️ **{quality}** - Moderate quality recovery")
                        else:
                            st.error(f"❌ **{quality}** - Low quality recovery")

                        # Display metrics in columns
                        st.markdown("## 📊 Quality Metrics")
                        col1, col2, col3 = st.columns(3)

                        # MSE
                        with col1:
                            mse_val = metrics['mse']
                            mse_color, mse_interp = get_metric_interpretation('mse', mse_val)

                            st.markdown(f"### MSE")
                            st.markdown(f"<h1 style='text-align: center; color: {mse_color};'>{mse_val:.2f}</h1>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>{mse_interp}</p>", 
                                      unsafe_allow_html=True)
                            st.caption("Lower is better (0 = perfect)")

                        # PSNR
                        with col2:
                            psnr_val = metrics['psnr']
                            psnr_color, psnr_interp = get_metric_interpretation('psnr', psnr_val)

                            st.markdown(f"### PSNR")
                            if psnr_val == float('inf'):
                                st.markdown(f"<h1 style='text-align: center; color: {psnr_color};'>∞ dB</h1>", 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h1 style='text-align: center; color: {psnr_color};'>{psnr_val:.2f} dB</h1>", 
                                          unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>{psnr_interp}</p>", 
                                      unsafe_allow_html=True)
                            st.caption("Higher is better (>30 dB = good)")

                        # SSIM
                        with col3:
                            ssim_val = metrics['ssim']
                            ssim_color, ssim_interp = get_metric_interpretation('ssim', ssim_val)

                            st.markdown(f"### SSIM")
                            st.markdown(f"<h1 style='text-align: center; color: {ssim_color};'>{ssim_val:.4f}</h1>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>{ssim_interp}</p>", 
                                      unsafe_allow_html=True)
                            st.caption("Higher is better (1 = perfect)")

                        # Metric interpretation table
                        st.markdown("---")
                        st.markdown("## 📖 Metric Interpretation Guide")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("**MSE (Mean Squared Error)**")
                            st.markdown("""
                            - 🟢 **< 100**: Excellent
                            - 🟡 **100-1000**: Good to Fair
                            - 🔴 **> 1000**: Poor
                            """)

                        with col2:
                            st.markdown("**PSNR (dB)**")
                            st.markdown("""
                            - 🟢 **> 40 dB**: Excellent
                            - 🟢 **30-40 dB**: Good
                            - 🟡 **20-30 dB**: Fair
                            - 🔴 **< 20 dB**: Poor
                            """)

                        with col3:
                            st.markdown("**SSIM**")
                            st.markdown("""
                            - 🟢 **> 0.95**: Excellent
                            - 🟢 **0.85-0.95**: Good
                            - 🟡 **0.7-0.85**: Fair
                            - 🔴 **< 0.7**: Poor
                            """)

                        # Visual difference map
                        with st.expander("🔬 View Difference Map (Visual Comparison)", expanded=True):
                            diff_map = calculate_difference_map(img1, img2)

                            st.markdown("### Pixel-Level Differences (Enhanced 5x)")
                            st.markdown("*Brighter areas indicate larger differences between images*")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown("**Original**")
                                st.image(img1, use_container_width=True, clamp=True)

                            with col2:
                                st.markdown("**Recovered**")
                                st.image(img2, use_container_width=True, clamp=True)

                            with col3:
                                st.markdown("**Difference Map**")
                                # Apply red colormap for better visualization
                                if len(diff_map.shape) == 2:
                                    diff_colored = cv2.applyColorMap(diff_map, cv2.COLORMAP_HOT)
                                    st.image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB), 
                                           use_container_width=True, clamp=True)
                                else:
                                    st.image(diff_map, use_container_width=True, clamp=True)

                except Exception as e:
                    st.error(f"❌ Error during comparison: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    else:
        st.info("👆 Please upload or select both images to compare")