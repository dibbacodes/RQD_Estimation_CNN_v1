import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
from torchvision import transforms
from PIL import Image

from CNN_model import Net, load_model
from RQD_calculation import calculate_rqd
from image_processing import (
    main_extract,
    get_depth,
    get_scales,
    get_header_depths,
    crop_outer_tray_box,
    split_by_profiles,
    generate_ssi_images
)

st.title("🪨 Rock Quality Designation (RQD) Estimator")

uploaded_file = st.file_uploader("Upload a core tray image", type=['jpg', 'jpeg', 'png'])


# Reset button appears after upload
if uploaded_file:
    if st.button("🔄 Reset"):
        st.experimental_rerun()

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 0: Save uploaded file to temp dir
        image_path = os.path.join(tmpdir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = cv2.imread(image_path)

        if image is not None:
            with st.spinner("Displaying Image..."):
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Core Tray", use_column_width=True)

            # Step 1: Preprocessing and Depth Extraction
            with st.spinner("Extracting depth and scale information..."):
                extracted, script_vert, script_h, script_L_D, r1, r2 = main_extract(image)
                depths = get_depth(script_vert, script_L_D)
                scale = get_scales(script_h)
                st.write("**Extracted Depths (in meters):** " + ", ".join(depths))
                st.write(f"**Pixels per 100 mm (scale):** {scale}")

            # Step 2: Header Depths
            with st.spinner("Extracting header depths..."):
                start_depth, end_depth = get_header_depths(image)
                if start_depth and end_depth:
                    st.write(f"**Start Depth:** {start_depth} m and **End Depth** {end_depth} m")
                else:
                    st.warning("Could not detect depth range from header.")

            # Step 3: Crop Outer Tray Box
            with st.spinner("Cropping tray from image..."):
                cropped_path = os.path.join(tmpdir, "final_cropped_tray.png")
                crop_outer_tray_box(image_path, output_path=cropped_path)

            # Step 4: Split into Tray Rows
            with st.spinner("Splitting tray into rows..."):
                tray_split_dir = os.path.join(tmpdir, "tray_split_by_profile")
                os.makedirs(tray_split_dir, exist_ok=True)
                split_by_profiles(input_path=cropped_path, output_dir=tray_split_dir)

            # Step 5: Generate Sliding Window Patches
            with st.spinner("Generating image patches for analysis..."):
                ssi_output_dir = os.path.join(tmpdir, "ssi_output")
                os.makedirs(ssi_output_dir, exist_ok=True)
                generate_ssi_images(input_dir=tray_split_dir, output_dir=ssi_output_dir, overlap_ratio=0.6)

            # Step 6: Load Trained CNN Model
            with st.spinner("Loading trained model..."):
                model_path = 'models/trained_model.pt'
                class_to_idx_path = 'models/class_to_idx.pth'
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model, class_to_idx = load_model(model_path, class_to_idx_path, model=Net(), device=device)

            # Step 7: RQD Calculation
            with st.spinner("Calculating RQD..."):
                rqd = calculate_rqd(
                    image_dir=ssi_output_dir,
                    model=model,
                    class_to_idx=class_to_idx,
                    scale=scale,
                    depths=depths,
                    start_depth=start_depth,
                    end_depth=end_depth
                )
                st.success(f"Estimated RQD: {rqd:.2f}%")

        else:
            st.error("Image could not be loaded. Please check the file.")
