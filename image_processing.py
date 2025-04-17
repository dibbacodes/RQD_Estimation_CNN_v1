import cv2
import numpy as np
import PIL
from PIL import Image
from pytesseract import image_to_string
import math
import os
from scipy.signal import find_peaks
import re

def main_extract(img):
    """
    Main function to extract regions of interest from the core tray image.
    
    Parameters:
    img (numpy array): Input image array (BGR format).
    
    Returns:
    tuple: Extracted regions (extracted, script_vert, script_h, script_L_D, r1, r2).
    """
    ht, wt, dt = img.shape
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding: invert binary image
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological closing to clean the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the morphed image
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and select the largest one
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Extract different regions from the image
    extracted = img[y:y + h, x:x + w]           # Extracted region (bounding box)
    script_vert = img[y:y + h, 0:x]              # Vertical script region
    shift_up = int(math.sqrt((x - x) ** 2 + (ht - (y + h)) ** 2) / 2)  # Calculate upward shift
    script_h = img[y + h - shift_up:ht - shift_up, x:x + w]  # Horizontal script region
    script_L_D = img[0:y, x:w]  # Lower depth section
    
    r1 = x  # Right boundary
    r2 = y  # Bottom boundary
    
    return (extracted, script_vert, script_h, script_L_D, r1, r2)

def isdigit(d):
    """
    Checks if a string can be converted to a float.

    Parameters:
    d (str): The string to check.

    Returns:
    bool: True if the string can be converted to a float, else False.
    """
    try:
        float(d)
        return True
    except ValueError:
        return False

def get_depth(script_vert, script_L_D):
    """
    Extracts depth information from vertical script region and lower depth section.
    
    Parameters:
    script_vert (numpy array): Vertical script region of the image.
    script_L_D (numpy array): Lower depth section of the image.
    
    Returns:
    list: List of depths found in the image.
    """
    # Extract and process depth information from the vertical script region
    script_vert = script_vert[0:script_vert.shape[0], int(0.5 * script_vert.shape[1]): script_vert.shape[1]]
    depths0 = Image.fromarray(script_vert)
    depths0 = depths0.transpose(PIL.Image.ROTATE_270)
    
    # Create a new image for depth extraction
    new = Image.new(mode=depths0.mode, size=(depths0.width, 2 * depths0.height), color='white')
    new.paste(depths0)
    
    # Resize and apply OCR to extract depth values
    depths1 = new.resize((int(0.3 * new.width), int(0.3 * new.height)))
    depths = image_to_string(depths1, config='-c tessedit_char_whitelist=.0123456789m')
    depths = depths.replace('m', ' ')
    depths = depths.split(' ')
    
    # Filter and sort depths
    depths_list = [d for d in depths if isdigit(d)]
    
    # Extract additional depth info from the lower depth section
    last_depth = Image.fromarray(script_L_D)
    last_depth = image_to_string(last_depth, config='-c tessedit_char_whitelist=to.0123456789m')
    last_depth = last_depth.split('to')[-1]
    last_depth = last_depth.split('m')[0]
    
    if isdigit(last_depth):
        depths_list.append(last_depth)
    
    # Sort the depths
    depths_list = sorted(depths_list, key=float)
    
    return depths_list

def get_scales(script_h):
    """
    Extracts scale from the horizontal script region using line detection and OCR.
    
    Parameters:
    script_h (numpy array): Horizontal script region of the image.
    
    Returns:
    int: The scale factor (pixels per 10 cm).
    """
    # Convert to grayscale and threshold the image
    gray = cv2.cvtColor(script_h, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean the image
    kern = np.ones((1, int(0.5 * script_h.shape[1])), np.uint8)
    opening = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kern)
    
    # Detect horizontal lines using Hough transform
    minLineLength = int(0.8 * script_h.shape[1])
    maxLineGap = 0
    lines = cv2.HoughLinesP(opening, 1, np.pi / 180, minLineLength, minLineLength, maxLineGap)
    
    # Find the y-coordinate of the top line
    y_list = sorted([lines[i, 0, 1] for i in range(len(lines))], key=int, reverse=True)
    y = y_list[0]
    
    # Extract a section of the image containing the scale
    scale_image = threshed[y:y + int(script_h.shape[0] / 3), 0:script_h.shape[1]]
    
    # Perform morphological operations to refine the scale image
    kern = np.ones((int(0.5 * scale_image.shape[0]), 1), np.uint8)
    eroded = cv2.erode(scale_image, kern, iterations=2)
    kernd = np.ones((scale_image.shape[0], 1), np.uint8)
    dilated = cv2.dilate(eroded, kernd, iterations=2)
    
    # Detect vertical lines for scaling
    minLineLength = int(0.5 * dilated.shape[0])
    maxLineGap = 2
    lines_v = cv2.HoughLinesP(dilated, 1, np.pi / 180, minLineLength, minLineLength, maxLineGap)
    
    # Calculate the pixel distance between scale markers
    line_list = sorted([lines_v[i, 0, 0] for i in range(len(lines_v))], key=int)
    distance_pixels = line_list[1] - line_list[0]
    
    # Extract the scale values using OCR
    scale1 = threshed[y:y + int(threshed.shape[0]), 0:script_h.shape[1]]
    kernel = np.ones((3, 3), np.uint8)
    scale2 = cv2.morphologyEx(scale1, cv2.MORPH_OPEN, kernel)
    scale2 = cv2.bitwise_not(scale2)
    scale_img = Image.fromarray(scale2)
    scale_list = image_to_string(scale_img, lang='eng', config='--psm 7 -c tessedit_char_whitelist=0123456789')
    scale_list = scale_list.split(' ')
    scale_list = [s for s in scale_list if s.isdigit()]
    
    # Sort and compute pixel equivalent for 10 cm
    scale_list = sorted(scale_list, key=int)
    distance_mm = int(scale_list[2]) - int(scale_list[1])
    pixels_eq_10cm = 100 * distance_pixels / distance_mm
    
    return int(pixels_eq_10cm)

def get_header_depths(img):
    """
    Extracts the start and end depths from the header section of the image.
    
    Parameters:
    img (numpy array): Input image (BGR format).
    
    Returns:
    tuple: Start and end depths (start_depth, end_depth).
    """
    # Crop the top portion (adjust if needed)
    header_crop = img[0:int(img.shape[0] * 0.1), :]  # Top 10% of image
    header_pil = Image.fromarray(header_crop)
    
    # OCR to extract text
    header_text = image_to_string(header_pil)
    
    # Search for depth range like "172.4 to 177 m"
    match = re.search(r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*m', header_text)
    if match:
        start_depth = float(match.group(1))
        end_depth = float(match.group(2))
        return start_depth, end_depth
    else:
        return None, None

##############################################################################
############## IMAGE CROPPING ##############################

def crop_outer_tray_box(input_path, output_path="final_cropped_tray.png"):
    """
    Crops the outer tray box from the core tray image.
    
    Parameters:
    input_path (str): Path to the input image.
    output_path (str): Path to save the cropped image.
    """
    # Load image and convert to grayscale
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and Canny edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours and get the bounding box of the largest contour
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda b: b[2] * b[3])
    
    # Crop the image using the bounding box
    cropped = image[y:y + h, x:x + w]
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_crop, 240, 255, cv2.THRESH_BINARY)
    
    # Remove unnecessary rows
    row_sums = np.sum(binary == 255, axis=1)
    content_rows = np.where(row_sums < binary.shape[1] * 0.98)[0]
    if len(content_rows) > 0:
        top, bottom = content_rows[0], content_rows[-1]
        cleaned = cropped[top:bottom + 1, :]
    else:
        cleaned = cropped
    
    # Save the cleaned cropped tray
    cv2.imwrite(output_path, cleaned)
    print(f"✅ Final cleaned tray saved to: {output_path}")

def split_by_profiles(input_path: str, output_dir: str = "tray_split_by_profile") -> None:
    """
    Splits a cleaned core tray image into multiple tray sections using detected horizontal dividers.

    Args:
        input_path (str): Path to the cleaned tray image.
        output_dir (str): Directory to save the split tray sections. Default is 'tray_split_by_profile'.

    Saves:
        Five cropped tray images (tray_1.png to tray_5.png) in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Error: Could not read image from {input_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute mean intensity per row
    row_means = np.mean(gray, axis=1)

    # Detect peaks in brightness (assumed tray dividers)
    peaks, _ = find_peaks(row_means, distance=gray.shape[0]//6, prominence=10)

    # Sort and select 4 divider positions
    divider_lines = sorted(peaks[:4])
    bounds = [0] + divider_lines + [gray.shape[0]]

    # Crop and save trays
    for i in range(5):
        y_start = bounds[i]
        y_end = bounds[i+1]
        tray = image[y_start:y_end, :]
        output_path = os.path.join(output_dir, f"tray_{i+1}.png")
        cv2.imwrite(output_path, tray)

    print(f"✅ Trays saved to: {output_dir}")


def generate_ssi_images(input_dir, output_dir, overlap_ratio=0.6):
    """
    Extracts sliding window image patches (SSIs) from images in the input directory and saves them to the output directory.
    
    Parameters:
    input_dir (str): Directory containing the input images (core tray images).
    output_dir (str): Directory where the extracted SSIs will be saved.
    overlap_ratio (float): The overlap ratio between consecutive patches (default 0.6).
    """
    os.makedirs(output_dir, exist_ok=True)
    ssi_count = 0

    for tray_file in sorted(os.listdir(input_dir)):
        tray_path = os.path.join(input_dir, tray_file)
        tray = cv2.imread(tray_path)
        height, width = tray.shape[:2]
        
        # Use tray height as SSI size (square window)
        win_size = height
        stride = int(win_size * (1 - overlap_ratio))

        for x in range(0, width - win_size + 1, stride):
            ssi = tray[:, x:x+win_size]  # Full height, sliding in width
            ssi_filename = f"ssi_{ssi_count:04d}.png"
            cv2.imwrite(os.path.join(output_dir, ssi_filename), ssi)
            ssi_count += 1

    print(f"✅ Extracted {ssi_count} SSIs to: {output_dir}")