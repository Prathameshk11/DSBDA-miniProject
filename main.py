import cv2
import numpy as np

def preprocess_image(image):
    """
    Convert to grayscale, blur, and binarize the image.
    Adaptive thresholding is used for robust thresholding under different lighting.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to smooth out noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding for variable lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   15, 10)
    return thresh

import matplotlib.pyplot as plt

def show_image_with_bbox(image, bbox):
    import cv2
    cv2.rectangle(image, (bbox[0], bbox[1]), 
                  (bbox[0]+bbox[2], bbox[1]+bbox[3]), 
                  (0,255,0), 2)

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image_rgb)
    plt.title("Detected Area")
    plt.axis('off')
    plt.show()


def get_written_area(thresh, size=(1240, 1754)):
    """
    Find contours and select the largest bounding rectangle which presumably
    corresponds to the handwritten area. 
    Use morphology to merge nearby components if needed.
    """
    # Use dilation to merge individual handwriting strokes if necessary.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours on the dilated image.
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Select the largest contour by area, assuming handwriting dominates.
     # Compute bounding boxes for all valid contours
    img_area = size[0] * size[1]
    W, H = size
    valid_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if not (0.0002 * img_area < area < 0.04 * img_area):
            continue

        too_close_left = x < 0.02 * W
        too_close_bottom = (y + h) > 0.98 * H
        aspect_ratio = h / (w + 1e-5)

        # Reject small blobs on the border or weird tall vertical ones
        if (too_close_left or too_close_bottom) and area < 800:
            continue
        if too_close_left and aspect_ratio > 5 and area > 100:
            continue  # likely left-side shading or scan artifact Relative to image dimensions
        

        valid_boxes.append((x, y, w, h))


    if not valid_boxes:
        return None

    # Find extremes of all boxes
    x_vals = [x for x, y, w, h in valid_boxes]
    y_vals = [y for x, y, w, h in valid_boxes]
    x2_vals = [x + w for x, y, w, h in valid_boxes]
    y2_vals = [y + h for x, y, w, h in valid_boxes]

    # Unified bounding box
    x_min = min(x_vals)
    y_min = min(y_vals)
    x_max = max(x2_vals)
    y_max = max(y2_vals)

    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    return bbox

def analyze_margins(image, bounding_box, margin_threshold=50):
    """
    Calculates margin distances from the bounding box to page edges.
    Compares each margin with a threshold (in pixels) to determine if the margin is 'good' (i.e., sufficiently large).
    Adjust margin_threshold according to resolution.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = bounding_box
    
    # Calculate distances
    left_margin = x
    top_margin = y
    right_margin = img_w - (x + w)
    bottom_margin = img_h - (y + h)
    
    # Evaluate if each margin is above the threshold.
    left_good   = left_margin >= margin_threshold
    right_good  = right_margin >= margin_threshold
    top_good    = top_margin >= margin_threshold
    bottom_good = bottom_margin >= margin_threshold
    
    return [left_good, right_good, top_good, bottom_good], (left_margin, top_margin, right_margin, bottom_margin)

def analyze_line_orientation(thresh, debug=False):
    """
    Use probabilistic Hough Line Transform to detect lines in the threshold image.
    Analyze the angles of the detected lines to decide if lines are straight, sloped, or curved.
    This is a simplistic method:
        - if the standard deviation of angles is below a small value, lines are considered 'straight' or consistently 'sloped'.
        - if the mean angle is near zero, they are straight.
        - if the mean angle is significantly non-zero, they are sloped.
        - if the variation is high, it suggests curvature or irregularity.
    Returns a tuple of booleans: (is_line_straight, is_line_sloped, is_line_curved).
    """
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    # Using HoughLinesP to get line segments.
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=thresh.shape[1]//2, maxLineGap=20)
    
    if lines is None:
        # No lines detected; might mean free-form handwritten text.
        return False, False, False

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Avoid division by zero.
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
        angles.append(angle)
        if debug:
            pass
            # print(f"Line: ({x1}, {y1}), ({x2}, {y2}) angle: {angle:.2f}")

    angles = np.array(angles)
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    
    # Define thresholds (these may need tuning)
    angle_straight_threshold = 5.0  # if mean angle is within 5° of horizontal, count as straight
    std_angle_threshold = 10.0      # low variance implies consistency

    is_line_straight = (abs(mean_angle) <= angle_straight_threshold and std_angle <= std_angle_threshold)
    is_line_sloped   = (abs(mean_angle) > angle_straight_threshold and std_angle <= std_angle_threshold)
    is_line_curved   = std_angle > std_angle_threshold

    return is_line_straight, is_line_sloped, is_line_curved

def process_page_image(image_path, margin_threshold=50, debug=False):
    """
    Main function to process a page image and output a list of boolean results.
    The result list order:
        [left_margin_good, right_margin_good, top_margin_good, bottom_margin_good,
         is_line_straight, is_line_sloped, is_line_curved]
    """
    STANDARD_SIZE = (1240, 1754)  # A4 in pixels at ~150 DPI

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image. Check the path.")
    
    image = cv2.resize(image, STANDARD_SIZE, interpolation=cv2.INTER_AREA)


    # Preprocess the image to obtain a binarized version highlighting the writing.
    thresh = preprocess_image(image)
    
    # Get bounding box for written area.
    bbox = get_written_area(thresh)
    
    show_image_with_bbox(image,bbox)
    
    if bbox is None:
        # No writing detected; return defaults.
        margin_bools = (False, False, False, False)
    else:
        margin_bools, margins = analyze_margins(image, bbox, margin_threshold)
        if debug:
            print(f"Bounding box: {bbox}")
            print(f"Margins (L, T, R, B): {margins}")

    # Analyze line orientation
    # We use the thresholded image again; note: for lined paper you might wish to remove printed lines first.
    line_bools = analyze_line_orientation(thresh, debug=debug)
    if debug:
        pass
        # print(f"Line orientation booleans (straight, sloped, curved): {line_bools}")

    # Combine results
    result = list(margin_bools) + list(line_bools)
    return result

def analyze_personalities(result):
    personality = ""
    left_margin_good, right_margin_good, top_margin_good, bottom_margin_good, is_line_straight, is_line_sloped, is_line_curved = result
    if(not left_margin_good):
        personality += "Attachec to Family,People and culture, Unlikely to prefer settling outside their own country, stay in same job for long time, "
    else:
        personality += "Socially concious, Weak emotional bond with family, "
    
    if(right_margin_good):
        personality += "Fear of Failure, Follower mentality, may experience stagnation in long run, "
    else:
        personality += "Blind Risk Taker, Impulsive approach towards goals and opportunities, Adventurous, "
    
    if(is_line_sloped):
        personality += "Positive Energy, "
    
    if(bottom_margin_good):
        personality += "Productive, Effective in utilizing their time and resources, Strong vision for future, "
    else:
        personality += "Acts without hesitation, Tend to plan every moment of their lives"
        
    return personality
    
        

# Example usage:
if __name__ == '__main__':
    # Replace 'page_image.jpg' with your image file path.
    image_path = 'a01-049u.png'
    try:
        results = process_page_image(image_path, margin_threshold=50, debug=True)
        print("Results (left_margin_good, right_margin_good, top_margin_good, bottom_margin_good,"
              " is_line_straight, is_line_sloped, is_line_curved):")
        print(results)
        
        print("Personality assensment: ", analyze_personalities(results) )
    except Exception as e:
        print("Error:", e)
