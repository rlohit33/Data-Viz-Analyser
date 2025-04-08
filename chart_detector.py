import numpy as np
from PIL import Image
import cv2
from skimage import feature, color, measure

def detect_chart_type(image):
    """
    Detect the type of chart from an image.
    
    Args:
        image (PIL.Image): The input chart image
        
    Returns:
        str: Detected chart type ('bar', 'line', 'pie', 'scatter', etc.)
    """
    # Convert PIL image to numpy array for OpenCV processing
    try:
        img_array = np.array(image)
        
        # Check if image array is valid
        if img_array.size == 0:
            # Return a default value if image is empty
            print("Warning: Empty image detected")
            return "unknown"
        
        # Handle images with alpha channel (RGBA)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Convert RGBA to RGB by removing alpha channel
            print("Converting RGBA image to RGB")
            img_array = img_array[:, :, :3]
            
        # Convert to grayscale if the image is in color
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Ensure gray image is in the correct format (uint8)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # Get image dimensions
        height, width = gray.shape[:2]
        
        # Enhanced edge detection with more sensitive parameters
        # Lower thresholds to detect more edges, especially in low-contrast charts
        edges = cv2.Canny(gray, 40, 130)
        
        # Detect circles (for pie charts)
        # Add try-except to handle potential errors
        try:
            # Use extremely strict parameters for circle detection to avoid false positives
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=max(width, height) // 2,  # Even larger minimum distance
                param1=200,                        # Much more sensitivity to edge detection
                param2=75,                         # Much higher threshold for circle detection
                minRadius=min(width, height) // 4, # Even larger minimum radius
                maxRadius=min(width, height) // 2
            )
            
            # If circles are detected, we need to confirm it's a pie chart
            # by checking additional properties
            if circles is not None and len(circles) > 0:
                # Count the first 3 circles only, to avoid false positives
                circle_count = min(len(circles[0]), 3)
                print(f"DEBUG - Found {circle_count} potential pie chart circles")
                
                # Additional validation for pie charts:
                # Check if circle is positioned near the center of the image
                true_circles = 0
                for i in range(circle_count):
                    x, y, r = circles[0][i]
                    # Check if the circle is very near the center of the image
                    if (abs(x - width/2) < width/6 and 
                        abs(y - height/2) < height/6):
                        true_circles += 1
                
                if true_circles == 0:
                    print("DEBUG - No valid centered circles found, not a pie chart")
                    circles = None
                else:
                    print(f"DEBUG - Confirmed {true_circles} valid pie chart circles")
        except cv2.error:
            # If HoughCircles fails, set circles to None
            print("Warning: HoughCircles detection failed")
            circles = None
    except Exception as e:
        print(f"Error processing image: {e}")
        return "unknown"
    
    # Detect lines (for bar and line charts)
    # Use more sensitive line detection to ensure we catch all vertical bars
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=40,  # Lower threshold to detect more lines 
        minLineLength=width//12,  # Shorter minimum line length to detect smaller bars
        maxLineGap=15  # Larger max gap to connect nearby line segments
    )
    
    # Feature calculation for chart type detection
    horizontal_lines = 0
    vertical_lines = 0
    diagonal_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle to determine line orientation
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Categorize lines as horizontal, vertical, or diagonal
            # Use wider angle ranges to catch more vertical and horizontal lines
            if angle < 15 or angle > 165:
                horizontal_lines += 1
            elif 75 < angle < 105:  # Expanded vertical angle range
                vertical_lines += 1
            else:
                diagonal_lines += 1
    
    # Check for scatter plot characteristics
    try:
        # Ensure the image is in the correct format for blob detection
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Convert the image to grayscale for blob detection
            img_gray = color.rgb2gray(img_array)
            blobs = feature.blob_dog(img_gray, min_sigma=3, max_sigma=7, threshold=0.1)
            num_blobs = len(blobs)
        else:
            # Handle grayscale or other image formats
            print("Using grayscale image for blob detection")
            blobs = feature.blob_dog(gray, min_sigma=3, max_sigma=7, threshold=0.1)
            num_blobs = len(blobs)
    except Exception as e:
        print(f"Error in blob detection: {e}")
        blobs = []
        num_blobs = 0
    
    # Debug prints for chart detection
    print(f"DEBUG - Chart Detection: circles={circles is not None and len(circles) > 0}, blobs={num_blobs}, horizontal={horizontal_lines}, vertical={vertical_lines}, diagonal={diagonal_lines}")
    
    # Determine chart type based on detected features
    chart_type = None
    
    # Calculate more sophisticated detection features with stronger criteria
    is_bar_chart = vertical_lines >= 5 and vertical_lines > horizontal_lines * 0.5
    is_line_chart = horizontal_lines >= 5 and horizontal_lines > vertical_lines * 1.5 and diagonal_lines < horizontal_lines * 0.8
    is_scatter_plot = num_blobs > 20 and vertical_lines + horizontal_lines < num_blobs * 0.7
    
    # Calculate a pie chart confidence score (0-10) based on multiple factors
    pie_confidence = 0
    
    # Only consider pie confidence if circles were detected
    if circles is not None and len(circles) > 0:
        # Base confidence from valid circle detection
        num_valid_circles = sum(1 for circle in circles[0] if 
                               abs(circle[0] - width/2) < width/6 and 
                               abs(circle[1] - height/2) < height/6)
        
        # Only consider it a potential pie chart if at least one valid centered circle exists
        if num_valid_circles > 0:
            pie_confidence = 3  # Start with base confidence
            
            # Reduce confidence if there are many vertical or horizontal lines
            # (as pie charts typically don't have many straight lines)
            if vertical_lines > 10 or horizontal_lines > 10:
                pie_confidence -= 2
                
            # Reduce confidence if there are many blobs scattered around
            # (scatter plots have many blobs but pie charts have concentrated blobs)
            if num_blobs > 30:
                pie_confidence -= 1
                
            # Check for the ratio of vertical to horizontal lines
            # (pie charts should have roughly equal vertical/horizontal lines or few of both)
            vertical_horizontal_ratio = vertical_lines / max(1, horizontal_lines)
            if 0.7 <= vertical_horizontal_ratio <= 1.3:
                pie_confidence += 1
            elif vertical_horizontal_ratio > 2 or vertical_horizontal_ratio < 0.5:
                pie_confidence -= 1
    
    print(f"DEBUG - Chart type confidence scores: pie={pie_confidence}, bar={is_bar_chart}, line={is_line_chart}, scatter={is_scatter_plot}")
    
    # Prioritized decision logic with stricter pie chart confidence threshold
    if pie_confidence >= 4:
        # Only classify as pie chart if confidence is very high
        chart_type = "pie"
    elif is_bar_chart:
        # Strong evidence of a bar chart
        chart_type = "bar"
    elif is_line_chart:
        # Strong evidence of a line chart
        chart_type = "line"
    elif is_scatter_plot:
        # Strong evidence of a scatter plot
        chart_type = "scatter"
    elif diagonal_lines > horizontal_lines and diagonal_lines > vertical_lines:
        # Area charts have many diagonal lines
        chart_type = "area"
    else:
        # Default - use bar chart as it's the most common visualization type
        chart_type = "bar"
    
    print(f"DEBUG - Detected chart type: {chart_type}")
    return chart_type
