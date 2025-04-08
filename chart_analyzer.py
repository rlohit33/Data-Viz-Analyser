import numpy as np
from PIL import Image
import cv2
from skimage import color, feature
import matplotlib.pyplot as plt
from collections import Counter

def analyze_chart(image, chart_type):
    """
    Analyze a chart image and evaluate its quality based on visualization best practices.
    
    Args:
        image (PIL.Image): The input chart image
        chart_type (str): Type of the chart ('bar', 'line', 'pie', 'scatter', etc.)
        
    Returns:
        dict: Analysis results including scores and detected issues
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Check if we have an RGBA image and print information for debugging
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print(f"Input image is RGBA with shape: {img_array.shape}")
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        print(f"Input image is RGB with shape: {img_array.shape}")
    else:
        print(f"Input image has unexpected shape: {img_array.shape}")
    
    # Analysis results dictionary
    results = {
        'chart_type': chart_type,
        'scores': {},
        'issues': [],
        'overall_score': 0
    }
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # ---- Color Analysis ----
    color_score, color_issues = analyze_colors(img_array)
    results['scores']['color'] = color_score
    results['issues'].extend(color_issues)
    
    # ---- Composition Analysis ----
    composition_score, composition_issues = analyze_composition(img_array, chart_type)
    results['scores']['composition'] = composition_score
    results['issues'].extend(composition_issues)
    
    # ---- Readability Analysis ----
    readability_score, readability_issues = analyze_readability(img_array)
    results['scores']['readability'] = readability_score
    results['issues'].extend(readability_issues)
    
    # ---- Chart-specific Analysis ----
    if chart_type == 'bar':
        specific_score, specific_issues = analyze_bar_chart(img_array)
    elif chart_type == 'line':
        specific_score, specific_issues = analyze_line_chart(img_array)
    elif chart_type == 'pie':
        specific_score, specific_issues = analyze_pie_chart(img_array)
    elif chart_type == 'scatter':
        specific_score, specific_issues = analyze_scatter_chart(img_array)
    else:
        specific_score, specific_issues = 7, []
    
    results['scores']['chart_specific'] = specific_score
    results['issues'].extend(specific_issues)
    
    # Calculate overall score (average of all scores)
    scores = list(results['scores'].values())
    results['overall_score'] = round(sum(scores) / len(scores), 1)
    
    return results

def analyze_colors(img_array):
    """Analyze color usage and harmony in the chart"""
    # Handle RGBA images (with alpha channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in color analysis")
        img_array = img_array[:, :, :3]
    
    # Convert to RGB if not already
    if len(img_array.shape) == 2:
        return 5, ["Chart appears to be grayscale, consider adding color for better data differentiation"]
    
    # Ensure we have an RGB image
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return 5, ["Unsupported image format; ensure image is RGB"]
    
    # Convert to HSV for better color analysis
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Downsample the image for faster processing
    h, w = hsv_img.shape[:2]
    downsampled = hsv_img[::max(1, h//100), ::max(1, w//100), :]
    
    # Extract hue values for color variety analysis - reshape properly
    hues = downsampled[:, :, 0].reshape(-1)
    saturations = downsampled[:, :, 1].reshape(-1)
    values = downsampled[:, :, 2].reshape(-1)
    
    # Count colors (quantized)
    hue_quantized = (hues / 10).astype(int)
    color_counts = Counter(hue_quantized)
    dominant_colors = len([count for count in color_counts.values() if count > len(hue_quantized) * 0.05])
    
    # Check for color issues
    issues = []
    score = 7  # Default score
    
    # Check color variety
    if dominant_colors < 2:
        issues.append("Limited color variety detected; consider using more distinct colors")
        score -= 1
    elif dominant_colors > 7:
        issues.append("Too many colors may overwhelm viewers; consider reducing color palette")
        score -= 1
    
    # Check for low saturation
    if np.mean(saturations) < 50:
        issues.append("Colors appear to be low in saturation; consider more vibrant colors")
        score -= 1
    
    # Check for inappropriate contrast
    if np.std(values) < 30:
        issues.append("Low contrast between elements; consider increasing color contrast")
        score -= 1
    
    # Code for checking red-green combinations has been removed as requested
    
    return max(1, min(10, score)), issues

def analyze_composition(img_array, chart_type):
    """Analyze the overall composition of the chart"""
    # Initialize score and issues
    score = 8
    issues = []
    
    # Handle RGBA images (with alpha channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in composition analysis")
        img_array = img_array[:, :, :3]
    
    # Ensure we have an RGB image
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        issues.append("Unsupported image format; ensure image is RGB")
        score -= 2
        return max(1, min(10, score)), issues
    
    # Check for text content (likely labels, title, etc.)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Calculate ratio of edges to detect density and complexity
    edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Check for chart complexity
    if edge_ratio > 0.2:
        issues.append("Chart appears to be visually complex; consider simplifying")
        score -= 1
    
    # Chart-specific composition checks
    if chart_type == 'bar' or chart_type == 'line':
        # Check for grid lines
        horizontal_lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50, 
            minLineLength=img_array.shape[1]//3, 
            maxLineGap=10
        )
        
        if horizontal_lines is None or len(horizontal_lines) < 3:
            issues.append("Few or no grid lines detected; consider adding for better readability")
            score -= 1
    
    elif chart_type == 'pie':
        # Too many pie slices is hard to read
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=0, maxRadius=0
        )
        
        if circles is not None:
            # Use contours to estimate number of pie slices
            _, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 10:
                issues.append("Pie chart has many segments; consider using a bar chart for better clarity")
                score -= 2
    
    # Check for whitespace/margins
    non_white_pixels = np.sum(np.any(img_array < 240, axis=2))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    content_ratio = non_white_pixels / total_pixels
    
    if content_ratio > 0.9:
        issues.append("Chart appears crowded; consider adding more whitespace/margins")
        score -= 1
    
    return max(1, min(10, score)), issues

def analyze_readability(img_array):
    """Analyze readability aspects like text visibility, font sizes, etc."""
    score = 7
    issues = []
    
    # Handle RGBA images (with alpha channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in readability analysis")
        img_array = img_array[:, :, :3]
    
    # Ensure we have an RGB image
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        issues.append("Unsupported image format; ensure image is RGB")
        score -= 2
        return max(1, min(10, score)), issues
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use MSER (Maximally Stable Extremal Regions) to detect text regions
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    # If text regions could not be detected
    if len(regions) < 5:
        issues.append("Few or no text elements detected; ensure proper labeling")
        score -= 2
    
    # Check for text contrast by analyzing edge intensity in text regions
    if len(regions) > 0:
        # Create a mask of potential text regions
        text_mask = np.zeros_like(gray)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(text_mask, [hull], 0, 255, -1)
        
        # Calculate contrast in text regions
        if np.sum(text_mask > 0) > 0:  # Make sure we have text regions
            text_pixels = gray[text_mask > 0]
            background_pixels = gray[text_mask == 0]
            
            if len(text_pixels) > 0 and len(background_pixels) > 0:
                text_mean = np.mean(text_pixels)
                bg_mean = np.mean(background_pixels)
                contrast = abs(text_mean - bg_mean)
                
                if contrast < 50:
                    issues.append("Low text contrast detected; increase contrast between text and background")
                    score -= 1
    
    # Check for image sharpness/blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("Image appears to be blurry; ensure visualization is sharp and clear")
        score -= 1
    
    return max(1, min(10, score)), issues

def analyze_bar_chart(img_array):
    """Special analysis for bar charts"""
    score = 7
    issues = []
    
    # Handle RGBA images (with alpha channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in bar chart analysis")
        img_array = img_array[:, :, :3]
    
    # Ensure we have an RGB image
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        issues.append("Unsupported image format; ensure image is RGB")
        score -= 2
        return max(1, min(10, score)), issues
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect edges and vertical lines (bars)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50, 
        minLineLength=img_array.shape[0]//5, 
        maxLineGap=10
    )
    
    vertical_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 80 < angle < 100:  # Vertical lines
                vertical_lines += 1
    
    # Too many bars
    if vertical_lines > 15:
        issues.append("Too many bars detected; consider grouping categories or using a different chart type")
        score -= 2
    
    # Too few bars
    if vertical_lines < 3:
        issues.append("Few bars detected; ensure data is properly represented")
        score -= 1
    
    # Check for 3D effect (generally not recommended)
    hist_3d = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peaks = np.sum(hist_3d > np.mean(hist_3d) + 2 * np.std(hist_3d))
    
    if peaks > 15:  # Many peaks in histogram often indicate 3D effects with many shades
        issues.append("Possible 3D effect detected; consider using flat 2D bars for better data perception")
        score -= 2
    
    return max(1, min(10, score)), issues

def analyze_line_chart(img_array):
    """Special analysis for line charts"""
    score = 7
    issues = []
    
    # Handle RGBA images (with alpha channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in line chart analysis")
        img_array = img_array[:, :, :3]
    
    # Ensure we have an RGB image
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        issues.append("Unsupported image format; ensure image is RGB")
        score -= 2
        return max(1, min(10, score)), issues
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50, 
        minLineLength=img_array.shape[1]//8, 
        maxLineGap=20
    )
    
    # Count diagonal lines (potential trend lines)
    diagonal_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 20 < angle < 80 or 100 < angle < 160:  # Diagonal lines
                diagonal_lines += 1
    
    # Too many lines
    if diagonal_lines > 8:
        issues.append("Many lines detected; consider using multiple charts or highlighting key trends")
        score -= 2
    
    # Check for line intersections (potential clutter)
    if lines is not None and len(lines) > 5:
        intersection_count = 0
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # Simple line intersection check
                d = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
                if d != 0:  # Non-parallel lines
                    intersection_count += 1
        
        if intersection_count > 5:
            issues.append("Many line intersections detected; consider separating lines or using multiple charts")
            score -= 1
    
    return max(1, min(10, score)), issues

def analyze_pie_chart(img_array):
    """Special analysis for pie charts"""
    score = 7
    issues = []
    
    print("DEBUG - Starting pie chart analysis")
    
    # Handle RGBA images (with alpha channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in pie chart analysis")
        img_array = img_array[:, :, :3]
    
    # Ensure we have an RGB image
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        issues.append("Unsupported image format; ensure image is RGB")
        score -= 2
        print(f"DEBUG - Pie chart analysis completed with issues: {issues}")
        return max(1, min(10, score)), issues
    
    # Check image dimensions
    height, width = img_array.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use more restrictive circle detection for verification
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=max(width, height) // 3,
        param1=150, 
        param2=50, 
        minRadius=min(width, height) // 5, 
        maxRadius=min(width, height) // 2
    )
    
    # Verify it's actually a pie chart
    if circles is None or len(circles[0]) == 0:
        issues.append("No clear circular shape detected; ensure pie chart is properly formed or consider using a different chart type")
        score -= 2
        print(f"DEBUG - No circles found in pie chart analysis")
        print(f"DEBUG - Pie chart analysis completed with issues: {issues}")
        return max(1, min(10, score)), issues
    
    # Check if the main circle is near the center of the image
    center_circle_found = False
    for circle in circles[0]:
        x, y, r = circle
        if (abs(x - width/2) < width/4 and abs(y - height/2) < height/4):
            center_circle_found = True
            break
            
    if not center_circle_found:
        issues.append("Pie chart not centered in the image; consider reframing the visualization")
        score -= 1
        print(f"DEBUG - No centered circle found in pie chart analysis")
    
    # Use color segmentation to estimate number of pie slices
    # Downsample image for faster processing
    small_img = cv2.resize(img_array, (100, 100))
    # Now small_img should be RGB with shape (100, 100, 3)
    # Reshape correctly with number of pixels and 3 color channels
    pixels = small_img.reshape((small_img.shape[0] * small_img.shape[1], 3))
    pixels = np.float32(pixels)
    
    # k-means clustering to find color segments (potential pie slices)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 8  # Start with assumption of 8 possible colors/segments
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count unique segments (excluding background)
    segment_counts = Counter(labels.flatten())
    significant_segments = [count for color, count in segment_counts.items() 
                           if count > len(labels.flatten()) * 0.05]
    
    # Too many segments
    if len(significant_segments) > 7:
        issues.append("Pie chart has too many segments; consider using a bar chart or grouping small categories")
        score -= 2
    
    # Check for tiny slices
    if len(significant_segments) > 2:
        smallest_segment = min(significant_segments)
        largest_segment = max(significant_segments)
        
        if smallest_segment < largest_segment * 0.05:
            issues.append("Chart contains very small slices; consider grouping into 'Other' category")
            score -= 1
    
    print(f"DEBUG - Pie chart analysis completed with issues: {issues}")
    return max(1, min(10, score)), issues

def analyze_scatter_chart(img_array):
    """Special analysis for scatter plots"""
    score = 7
    issues = []
    
    # Handle RGBA images (with alpha channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in scatter chart analysis")
        img_array = img_array[:, :, :3]
    
    # Ensure we have an RGB image
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        issues.append("Unsupported image format; ensure image is RGB")
        score -= 2
        return max(1, min(10, score)), issues
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect blobs (scatter points)
    blobs = feature.blob_dog(color.rgb2gray(img_array), min_sigma=2, max_sigma=5, threshold=0.1)
    
    # Check number of points
    num_points = len(blobs)
    
    if num_points > 200:
        issues.append("Large number of scatter points may cause visual clutter; consider opacity or sampling")
        score -= 1
    
    if num_points < 10:
        issues.append("Few data points detected; scatter plots are most effective with more points")
        score -= 1
    
    # Check for axis lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50, 
        minLineLength=img_array.shape[1]//4, 
        maxLineGap=10
    )
    
    has_horizontal_axis = False
    has_vertical_axis = False
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal lines
                has_horizontal_axis = True
            elif 80 < angle < 100:  # Vertical lines
                has_vertical_axis = True
    
    if not (has_horizontal_axis and has_vertical_axis):
        issues.append("Axis lines not clearly detected; ensure clear X and Y axes for better readability")
        score -= 1
    
    # Check for point clustering
    if num_points > 20:
        # Simple analysis of point distribution
        x_coords = blobs[:, 1]
        y_coords = blobs[:, 0]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        # If points are too clustered or too uniform
        if x_std < img_array.shape[1] * 0.05 or y_std < img_array.shape[0] * 0.05:
            issues.append("Data points appear clustered; consider zooming in or using logarithmic scale")
            score -= 1
    
    return max(1, min(10, score)), issues
