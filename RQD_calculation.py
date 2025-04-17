import os
import glob
import cv2
from CNN_model import predict_image

# RQD Calculation Logic
def calculate_rqd(image_dir, model, class_to_idx, scale, depths, start_depth, end_depth, overlap_ratio=0.6):
    pixels_per_100mm = scale
    mm_per_pixel = 100 / pixels_per_100mm

    ssi_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    if depths and len(depths) >= 2:
        total_run_length_mm = (end_depth - start_depth) * 1000
    else:
        raise ValueError("Depths not extracted properly!")

    rqd_length_mm = 0
    temp = 0
    prev_whole = False
    
    print("Calculating RQD...")

    for ssi_path in ssi_paths:
        ssi_image = cv2.imread(ssi_path)
        predicted_class = predict_image(ssi_image, model, class_to_idx)

        length_px = ssi_image.shape[1]
        length_mm = length_px * mm_per_pixel

        if predicted_class == 'whole':
            if prev_whole:
                temp += length_mm * (1 - overlap_ratio)
            else:
                temp += length_mm
            prev_whole = True
        else:
            if temp >= 100:
                rqd_length_mm += temp
            temp = 0
            prev_whole = False

    # Final check for temp segment at the end
    if temp >= 100:
        rqd_length_mm += temp

    rqd_percent = (rqd_length_mm / total_run_length_mm) * 100 if total_run_length_mm > 0 else 0
    return rqd_percent
