"""
Home Page for Image Processing & Restoration System
Professional landing page with project overview and navigation.
"""

import streamlit as st


def render_home_page():
    """
    Render the home/landing page with project information and navigation cards.
    """
    
    # Hero section with title and description
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; color: #FF4B4B; margin-bottom: 0.5rem;'>
            🖼️ VisionAID
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Row 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; height: 200px;'>
            <h3 style='margin-top: 0;'>Upload & Degradation</h3>
            <p style='font-size: 0.9rem;'>
                Upload images and apply various degradation techniques like blur, noise, 
                and motion blur to simulate real-world scenarios.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; height: 200px;'>
            <h3 style='margin-top: 0;'>Non-Blind Deblurring</h3>
            <p style='font-size: 0.9rem;'>
                Restore blurred images when the blur kernel is known. 
                Includes Wiener and Inverse filtering techniques.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; height: 200px;'>
            <h3 style='margin-top: 0;'>Blind Deblurring</h3>
            <p style='font-size: 0.9rem;'>
                Advanced restoration when blur kernel is unknown. 
                Automatic blur estimation and removal.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; height: 200px;'>
            <h3 style='margin-top: 0;'>Denoising</h3>
            <p style='font-size: 0.9rem;'>
                Remove various types of noise using Mean, Median, Gaussian, 
                Bilateral, and Non-Local Means filters.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; height: 200px;'>
            <h3 style='margin-top: 0;'>Reduce Periodic Noise</h3>
            <p style='font-size: 0.9rem;'>
                Detect and remove periodic noise patterns using 
                Butterworth notch filters in frequency domain.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; height: 200px;'>
            <h3 style='margin-top: 0;'>Histogram Enhancement</h3>
            <p style='font-size: 0.9rem;'>
                Improve image contrast using histogram equalization 
                in grayscale, RGB, or HSV color spaces.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 3 (centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1.5rem; border-radius: 10px; color: #333; height: 200px;'>
            <h3 style='margin-top: 0;'>Visualization & Comparison</h3>
            <p style='font-size: 0.9rem;'>
                Compare original vs processed images using quality metrics: 
                MSE, PSNR, and SSIM with visual difference maps.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    