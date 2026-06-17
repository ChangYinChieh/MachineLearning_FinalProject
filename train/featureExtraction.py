import os
import cv2
import numpy as np

# ======================================================
# Path Settings
# ======================================================
INPUT_DIR = "./dataSet_test"                # Original root directory of the dataset
OUTPUT_DIR = "./dataSet_test_preprocessed"  # Destination directory for saving feature maps

# Supported image extensions
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

def process_single_image(img):
    """
    Core algorithm: Converts a single image into a high-quality, 
    smooth, and continuous gesture edge feature map.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # 1. Skin color detection and mask creation
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 170, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(skin_mask)
    img_h, img_w = skin_mask.shape[:2]
    img_center = np.array([img_w / 2, img_h / 2])
    img_area = img_h * img_w
    
    best_contour = None
    min_score = float('inf')
    
    # Smart central region filter
    for c in contours:
        area = cv2.contourArea(c)
        if area < (img_area * 0.02) or area > (img_area * 0.60):
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            if dist < min_score:
                min_score = dist
                best_contour = c
                
    if best_contour is not None:
        cv2.drawContours(hand_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    elif contours:
        best_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(hand_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    # 2. High-sensitivity Canny (applied to the raw image first to ensure smooth, natural lines)
    raw_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    raw_canny = cv2.Canny(raw_blurred, 15, 50)

    # 3. Expand the mask umbrella to filter out background noise
    mask_expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_hand_mask = cv2.dilate(hand_mask, mask_expand_kernel, iterations=1)
    filtered_edges = cv2.bitwise_and(raw_canny, raw_canny, mask=dilated_hand_mask)

    # 4. Morphology CLOSE operation to bridge small gaps, followed by a slight dilation
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(filtered_edges, cv2.MORPH_CLOSE, close_kernel)
    
    dilation_kernel = np.ones((3, 3), np.uint8)
    final_edges = cv2.dilate(closed_edges, dilation_kernel, iterations=1)
    
    # 5. Convert back to 3 channels (BGR) to match the input requirement of MiniVGGNet
    final_edges_3ch = cv2.cvtColor(final_edges, cv2.COLOR_GRAY2BGR)
    
    return final_edges_3ch

def main():
    print("========== Start Batch Preprocessing Gesture Dataset ==========")
    if not os.path.exists(INPUT_DIR):
        print(f"[Error] Source directory not found: {INPUT_DIR}. Please check your path settings.")
        return

    success_count = 0
    error_count = 0

    # Traverse all subdirectories and files in the source directory
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            # Check if the file matches supported image formats
            if file.lower().endswith(VALID_EXTENSIONS):
                # Construct the source file path
                src_path = os.path.join(root, file)
                
                # Calculate relative path to replicate the exact folder structure in the destination
                rel_path = os.path.relpath(root, INPUT_DIR)
                dest_dir = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                dest_path = os.path.join(dest_dir, file)
                
                # Read the image
                img = cv2.imread(src_path)
                if img is None:
                    print(f"[Warning] Unable to read file (Skipped): {src_path}")
                    error_count += 1
                    continue
                
                try:
                    # Execute the feature extraction algorithm
                    processed_img = process_single_image(img)
                    
                    # Save the result
                    cv2.imwrite(dest_path, processed_img)
                    success_count += 1
                    
                    # Print progress every 100 successfully processed images to avoid clutter
                    if success_count % 100 == 0:
                        print(f"[Progress] Successfully processed {success_count} images...")
                        
                except Exception as e:
                    print(f"[Error] Failed to process file: {src_path}, Reason: {str(e)}")
                    error_count += 1

    print("\n========== Batch Preprocessing Completed ==========")
    print(f"Successfully converted and saved: {success_count} images")
    print(f"Failed/Skipped images: {error_count}")
    print(f"Cleaned feature dataset saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()