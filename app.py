"""
Phase 11: Streamlit Web Interface
==================================
The main entry point for the AI-Powered CAD Comparison Tool.
Provides a simple, clean UI to upload drawings, run the pipeline,
and view the interactive visualizations and final report.

Run with: streamlit run app.py
"""

import os
import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image

# Pipeline modules
from app.input_handler import load_drawing_pair
from app.pixel_diff import run_pixel_diff
from app.yolo_detector import load_model, detect_on_pair
from app.ocr_engine import init_ocr, ocr_all_detections
from app.element_matcher import match_elements
from app.statistics import compute_statistics
from app.summarizer import generate_summary_report
from app.visualizer import create_combined_overlay, create_before_overlay, create_after_overlay

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CAD Comparison Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Helper: Caching models
# ---------------------------------------------------------------------------
@st.cache_resource
def get_models():
    """Load YOLO and OCR models once and cache them in memory."""
    print("[app] Loading models into cache...")
    model = load_model()
    ocr = init_ocr(use_gpu=False)  # CPU mode for stability
    return model, ocr

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------
def convert_cv2_to_pil(img_bgr):
    """Convert OpenCV BGR image to PIL RGB for Streamlit."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    st.title("🏗️ AI-Powered CAD Comparison Tool")
    st.markdown("Upload a 'Before' and 'After' CAD drawing to automatically detect, classify, and summarize structural and semantic changes.")

    # Sidebar for controls
    st.sidebar.header("1. Upload Drawings")
    
    file_a = st.sidebar.file_uploader("Upload 'Before' Drawing (v1)", type=['pdf', 'png', 'jpg', 'jpeg'])
    file_b = st.sidebar.file_uploader("Upload 'After' Drawing (v2)", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    st.sidebar.header("2. Settings")
    align_images = st.sidebar.checkbox("Auto-Align Images (ORB)", value=True, help="Fixes rotation and translation differences.")
    dpi = st.sidebar.slider("PDF Render DPI", min_value=150, max_value=400, value=300, step=50)

    # Initialize models
    with st.spinner("Loading AI Models..."):
        yolo_model, ocr_engine = get_models()

    if file_a and file_b:
        st.sidebar.success("Files uploaded successfully.")
        
        if st.sidebar.button("🚀 Run Comparison", type="primary"):
            
            # Create a progress container
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # --- Step 1: Input & Alignment ---
                status_text.text("Step 1/6: Loading and aligning drawings...")
                t0 = time.time()
                pair = load_drawing_pair(file_a.read(), file_b.read(), align=align_images, dpi=dpi)
                progress_bar.progress(20)

                # --- Step 2: Pixel-Level Diff ---
                status_text.text("Step 2/6: Computing pixel-level differences (SSIM)...")
                diff_res = run_pixel_diff(pair['gray_a'], pair['gray_b'], pair['img_a'], pair['img_b'])
                progress_bar.progress(40)

                # --- Step 3: YOLO Detection ---
                status_text.text("Step 3/6: Detecting dimensions and notes (YOLOv8)...")
                det_res = detect_on_pair(yolo_model, pair['img_a'], pair['img_b'])
                progress_bar.progress(60)

                # --- Step 4: OCR Extraction ---
                status_text.text("Step 4/6: Extracting text from regions (PaddleOCR)...")
                det_a = ocr_all_detections(ocr_engine, pair['img_a'], det_res['detections_a'])
                det_b = ocr_all_detections(ocr_engine, pair['img_b'], det_res['detections_b'])
                progress_bar.progress(75)

                # --- Step 5: Element Matching ---
                status_text.text("Step 5/6: Matching elements and classifying changes...")
                match_res = match_elements(det_a, det_b)
                progress_bar.progress(85)

                # --- Step 6: Statistics & Visualization ---
                status_text.text("Step 6/6: Generating final report and visualization...")
                stats = compute_statistics(diff_res, match_res)
                report = generate_summary_report(stats)
                
                final_overlay = create_combined_overlay(
                    image_b=pair['img_b'], 
                    match_results=match_res, 
                    image_a=pair['img_a'],
                    heatmap=diff_res['heatmap'],
                    heatmap_alpha=0.4
                )
                
                before_overlay = create_before_overlay(pair['img_a'], match_res)
                after_overlay = create_after_overlay(pair['img_b'], match_res)
                
                progress_bar.progress(100)
                elapsed = time.time() - t0
                status_text.text(f"✅ Pipeline completed in {elapsed:.1f} seconds.")
                
                # Save to session state so we can interact with UI widgets without losing data
                st.session_state['pair'] = pair
                st.session_state['diff_res'] = diff_res
                st.session_state['final_overlay'] = final_overlay
                st.session_state['before_overlay'] = before_overlay
                st.session_state['after_overlay'] = after_overlay
                st.session_state['report'] = report
                st.session_state['run_complete'] = True

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                progress_bar.empty()
                status_text.empty()
                
        # --- Display Results from Session State ---
        if st.session_state.get('run_complete', False):
            st.markdown("---")
            
            # Retrieve from session state
            pair = st.session_state['pair']
            diff_res = st.session_state['diff_res']
            final_overlay = st.session_state['final_overlay']
            before_overlay = st.session_state.get('before_overlay', pair['img_a'])
            after_overlay = st.session_state.get('after_overlay', pair['img_b'])
            report = st.session_state['report']
            
            col_vis, col_report = st.columns([6, 4])
            
            with col_vis:
                st.subheader("Visual Analysis")
                st.markdown("*Hint: Hover over the image and click the ⤢ icon in the top right to view fullscreen and zoom in.*")
                
                # Toggle between views
                view_option = st.radio(
                    "Select View Mode:",
                    (
                        "Combined Overlay (Heatmap + Semantic)", 
                        "Side-by-Side (Annotated)", 
                        "Annotated 'Before'", 
                        "Annotated 'After'", 
                        "Pure Heatmap",
                        "Raw Originals"
                    )
                )
                
                if view_option == "Combined Overlay (Heatmap + Semantic)":
                    st.image(convert_cv2_to_pil(final_overlay), use_container_width=True)
                    st.caption("🟢 Added | 🔴 Removed | 🟡 Modified | 🔵 Moved")
                elif view_option == "Side-by-Side (Annotated)":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(convert_cv2_to_pil(before_overlay), use_container_width=True, caption="Removed & Modified")
                    with col2:
                        st.image(convert_cv2_to_pil(after_overlay), use_container_width=True, caption="Added & Modified")
                elif view_option == "Annotated 'Before'":
                    st.image(convert_cv2_to_pil(before_overlay), use_container_width=True)
                    st.caption("🔴 Removed | 🟡 Modified | 🔵 Moved From Here")
                elif view_option == "Annotated 'After'":
                    st.image(convert_cv2_to_pil(after_overlay), use_container_width=True)
                    st.caption("🟢 Added | 🟡 Modified | 🔵 Moved To Here")
                elif view_option == "Pure Heatmap":
                    st.image(convert_cv2_to_pil(diff_res['heatmap']), use_container_width=True)
                elif view_option == "Raw Originals":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(convert_cv2_to_pil(pair['img_a']), use_container_width=True, caption="Raw Before")
                    with col2:
                        st.image(convert_cv2_to_pil(pair['img_b']), use_container_width=True, caption="Raw After")

            with col_report:
                st.subheader("Automated Report")
                with st.container(height=600, border=True):
                    st.markdown(report)
                    
                st.download_button(
                    label="📄 Download Report (Markdown)",
                    data=report,
                    file_name="cad_comparison_report.md",
                    mime="text/markdown"
                )
    else:
        st.info("👈 Please upload both CAD drawings in the sidebar to begin.")
        
        # Display sample workflow
        st.markdown("""
        ### How it works:
        1. **Pixel-Level Diff:** Identifies structural changes using SSIM and morphological analysis.
        2. **Semantic Detection:** Uses a trained YOLOv8 model to find dimensions and notes.
        3. **OCR Extraction:** Reads the text inside detected regions using PaddleOCR.
        4. **Element Matching:** Cross-references elements between drawings to classify changes as Added, Removed, Modified, or Moved.
        5. **AI Summarization:** Generates a clean, natural language report detailing every change.
        """)

if __name__ == "__main__":
    main()
