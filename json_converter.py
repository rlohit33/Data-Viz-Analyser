import numpy as np
from PIL import Image
import cv2
from skimage import feature, color, measure
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def visualization_to_json(image, chart_type):
    """
    Convert visualization image to JSON representation of the underlying data.
    
    Args:
        image (PIL.Image): The input chart image
        chart_type (str): Type of the chart ('bar', 'line', 'pie', 'scatter', etc.)
        
    Returns:
        dict: JSON representation of the visualization data
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Print image information for debugging
    print(f"Image shape in visualization_to_json: {img_array.shape}")
    
    # Handle RGBA images at top level before passing to specific extractors
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        print("Converting RGBA to RGB in visualization_to_json")
        img_array = img_array[:, :, :3]
    
    # Select appropriate extraction method based on chart type
    if chart_type == 'bar':
        extracted_data = extract_bar_chart_data(img_array)
    elif chart_type == 'line':
        extracted_data = extract_line_chart_data(img_array)
    elif chart_type == 'pie':
        extracted_data = extract_pie_chart_data(img_array)
    elif chart_type == 'scatter':
        extracted_data = extract_scatter_plot_data(img_array)
    else:
        # Default generic extraction for unsupported chart types
        extracted_data = extract_generic_chart_data(img_array, chart_type)
    
    # Add metadata to the JSON
    json_data = {
        'chart_type': chart_type,
        'extraction_method': 'computer_vision',
        'width': img_array.shape[1],
        'height': img_array.shape[0],
        'data': extracted_data
    }
    
    return json_data

def extract_bar_chart_data(img_array):
    """Extract data from a bar chart image"""
    try:
        # Handle RGBA images (with alpha channel)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            print("Converting RGBA to RGB in bar chart extraction")
            img_array = img_array[:, :, :3]
            
        # Ensure we have an RGB image
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img_array.shape}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"Error in bar chart extraction: {e}")
        # Return empty data as fallback
        return {
            'bars': [],
            'count': 0,
            'orientation': 'unknown',
            'extracted_values': []
        }
    
    # Threshold the image to separate bars from background
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (potential bars)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to identify bars
    bar_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small contours and those with odd aspect ratios
        if h > img_array.shape[0] * 0.05 and w > 5:  # Height at least 5% of image, width > 5px
            bar_contours.append((x, y, w, h))
    
    # Sort bars by x-coordinate (left to right)
    bar_contours.sort(key=lambda c: c[0])
    
    # Extract color information for each bar
    bar_data = []
    for i, (x, y, w, h) in enumerate(bar_contours):
        # Extract region of interest for the bar
        bar_roi = img_array[y:y+h, x:x+w]
        
        # Calculate average color of the bar
        avg_color = np.mean(bar_roi, axis=(0, 1))
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
        )
        
        # Calculate relative height (taller bar = higher value)
        # Inverted because image y-axis starts at top
        relative_height = h / img_array.shape[0]
        
        # Estimate value based on height
        # Assuming bars grow from bottom of chart
        value = relative_height * 100  # Scale to 0-100 for simplicity
        
        bar_data.append({
            'index': i,
            'x_position': x,
            'width': w,
            'height': h,
            'estimated_value': round(value, 2),
            'color': color_hex,
            'label': f"Category {i+1}"  # Placeholder label
        })
    
    return {
        'bars': bar_data,
        'count': len(bar_data),
        'orientation': 'vertical',  # Assumption, could be refined
        'extracted_values': [bar['estimated_value'] for bar in bar_data]
    }

def extract_line_chart_data(img_array):
    """Extract data from a line chart image"""
    try:
        # Handle RGBA images (with alpha channel)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            print("Converting RGBA to RGB in line chart extraction")
            img_array = img_array[:, :, :3]
            
        # Ensure we have an RGB image
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img_array.shape}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"Error in line chart extraction: {e}")
        # Return empty data as fallback
        return {
            'lines': [],
            'count': 0,
            'extracted_values': [],
            'error': str(e)
        }
    
    # Threshold the image to isolate lines
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Edge detection
    edges = cv2.Canny(threshold, 50, 150)
    
    # Line detection using Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50, 
        minLineLength=img_array.shape[1]//20, 
        maxLineGap=20
    )
    
    # Group lines by slope to identify potential trend lines
    line_groups = {}
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip perfectly horizontal or vertical lines (likely axes)
            if x1 == x2 or y1 == y2:
                continue
            
            # Calculate slope
            slope = (y2 - y1) / (x2 - x1)
            
            # Round slope to group similar lines
            rounded_slope = round(slope, 1)
            
            # Add to appropriate group
            if rounded_slope not in line_groups:
                line_groups[rounded_slope] = []
            
            line_groups[rounded_slope].append(line[0])
    
    # Assemble data for each potential trend line
    line_data = []
    for i, (slope, lines) in enumerate(line_groups.items()):
        # Average color along the line
        line_colors = []
        for x1, y1, x2, y2 in lines:
            # Sample points along the line
            num_samples = 10
            x_samples = np.linspace(x1, x2, num_samples).astype(int)
            y_samples = np.linspace(y1, y2, num_samples).astype(int)
            
            # Clip to ensure within image bounds
            x_samples = np.clip(x_samples, 0, img_array.shape[1]-1)
            y_samples = np.clip(y_samples, 0, img_array.shape[0]-1)
            
            # Extract colors
            for x, y in zip(x_samples, y_samples):
                if 0 <= y < img_array.shape[0] and 0 <= x < img_array.shape[1]:
                    line_colors.append(img_array[y, x])
        
        if line_colors:
            avg_color = np.mean(line_colors, axis=0)
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
            )
        else:
            color_hex = '#000000'  # Default to black if no colors sampled
        
        # Create points along the line path
        points = []
        if lines:
            # Sort lines by x-coordinate to get a continuous path
            sorted_lines = sorted(lines, key=lambda l: min(l[0], l[2]))
            
            # Extract points from each line segment
            for x1, y1, x2, y2 in sorted_lines:
                # Add more points for better representation
                step = max(1, abs(x2 - x1) // 10)
                for t in range(0, 11, step):
                    x = x1 + (t/10) * (x2 - x1)
                    y = y1 + (t/10) * (y2 - y1)
                    
                    # Convert y to a value (higher y = lower value since image origin is top-left)
                    # Normalize to 0-100 scale
                    normalized_y = 100 * (1 - y / img_array.shape[0])
                    
                    points.append({
                        'x': round(x, 1),
                        'y': round(y, 1),
                        'value': round(normalized_y, 2)
                    })
        
        # Only add if we have actual points
        if points:
            line_data.append({
                'id': i,
                'color': color_hex,
                'points': points,
                'point_count': len(points),
                'slope': slope,
                'label': f"Series {i+1}"  # Placeholder label
            })
    
    return {
        'lines': line_data,
        'count': len(line_data),
        'extracted_values': [[point['value'] for point in line['points']] for line in line_data]
    }

def extract_pie_chart_data(img_array):
    """Extract data from a pie chart image"""
    try:
        # Handle RGBA images (with alpha channel)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            print("Converting RGBA to RGB in pie chart extraction")
            img_array = img_array[:, :, :3]
        
        # Ensure we have an RGB image (3 channels)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img_array.shape}")
            
        # Use color segmentation to identify pie slices
        # Downsample image for faster processing
        downsampled = cv2.resize(img_array, (100, 100))
        
        # Reshape to a list of pixels (rows x cols, 3) - explicitly using shape dimensions
        pixels = downsampled.reshape((downsampled.shape[0] * downsampled.shape[1], 3))
        pixels = np.float32(pixels)
    except Exception as e:
        print(f"Error in pie chart extraction: {e}")
        # Return empty data as fallback
        return {
            'segments': [],
            'count': 0,
            'extracted_values': [],
            'error': str(e)
        }
    
    # Determine an appropriate number of clusters (pie slices)
    # Start with assumption of 5-8 segments
    # You could adapt this with a more sophisticated algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 8
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count pixels in each segment
    segment_counts = {}
    for i in range(k):
        segment_counts[i] = np.sum(labels == i)
    
    # Filter out very small segments (likely noise)
    total_pixels = len(labels)
    significant_segments = {i: count for i, count in segment_counts.items() 
                           if count > total_pixels * 0.03}  # At least 3% of total
    
    # Calculate segment percentages
    total_significant = sum(significant_segments.values())
    
    # Reconstruct the segmented image for visualization - be careful with shapes
    segmented_img = np.zeros_like(downsampled)
    
    # Reshape labels to match the image dimensions
    labels_reshaped = labels.reshape(downsampled.shape[0], downsampled.shape[1])
    
    # Assign colors to segments - using the reshaped labels
    for i, center in enumerate(centers):
        segmented_img[labels_reshaped == i] = center
    
    # No need to reshape since we're already working with the correct shape
    
    # Extract pie slice data
    pie_data = []
    for i, count in significant_segments.items():
        percentage = (count / total_significant) * 100
        
        # Get color of this segment
        color = centers[i]
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(color[0]), int(color[1]), int(color[2])
        )
        
        pie_data.append({
            'id': len(pie_data),
            'segment': f"Segment {len(pie_data)+1}",  # Placeholder label
            'percentage': round(percentage, 2),
            'color': color_hex,
            'pixel_count': int(count)
        })
    
    # Sort by percentage (largest first)
    pie_data.sort(key=lambda x: x['percentage'], reverse=True)
    
    return {
        'segments': pie_data,
        'count': len(pie_data),
        'extracted_values': [segment['percentage'] for segment in pie_data]
    }

def extract_scatter_plot_data(img_array):
    """Extract data from a scatter plot image"""
    try:
        # Handle RGBA images (with alpha channel)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            print("Converting RGBA to RGB in scatter plot extraction")
            img_array = img_array[:, :, :3]
            
        # Ensure we have an RGB image
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img_array.shape}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"Error in scatter plot extraction: {e}")
        # Return empty data as fallback
        return {
            'points': [],
            'count': 0,
            'x_values': [],
            'y_values': [],
            'error': str(e)
        }
    
    # Threshold to isolate points
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Use blob detection to find scatter points
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)
    
    # Alternative: use blob_dog from scikit-image
    if len(keypoints) < 5:  # If SimpleBlobDetector found few points, try blob_dog
        blobs = feature.blob_dog(color.rgb2gray(img_array), min_sigma=2, max_sigma=5, threshold=0.1)
        keypoints = []
        for blob in blobs:
            y, x, r = blob
            keypoints.append(cv2.KeyPoint(x=float(x), y=float(y), size=float(r)))
    
    # Extract point data
    point_data = []
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        
        # Ensure coordinates are within image bounds
        if 0 <= y < img_array.shape[0] and 0 <= x < img_array.shape[1]:
            # Get color at this point (average of small region around point)
            x1, y1 = max(0, x-3), max(0, y-3)
            x2, y2 = min(img_array.shape[1]-1, x+3), min(img_array.shape[0]-1, y+3)
            region = img_array[y1:y2, x1:x2]
            
            if region.size > 0:  # Make sure region is not empty
                avg_color = np.mean(region, axis=(0, 1))
                color_hex = '#{:02x}{:02x}{:02x}'.format(
                    int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
                )
            else:
                color_hex = '#000000'  # Default black if region is empty
            
            # Calculate normalized x and y values (0-100 scale)
            norm_x = 100 * (x / img_array.shape[1])
            # Invert y since image coordinates start at top-left
            norm_y = 100 * (1 - y / img_array.shape[0])
            
            point_data.append({
                'id': i,
                'x': float(x),
                'y': float(y),
                'normalized_x': round(norm_x, 2),
                'normalized_y': round(norm_y, 2),
                'color': color_hex,
                'size': keypoint.size
            })
    
    return {
        'points': point_data,
        'count': len(point_data),
        'x_values': [point['normalized_x'] for point in point_data],
        'y_values': [point['normalized_y'] for point in point_data]
    }

def extract_generic_chart_data(img_array, chart_type):
    """Generic data extraction for unsupported chart types"""
    try:
        # Handle RGBA images (with alpha channel)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            print("Converting RGBA to RGB in generic chart extraction")
            img_array = img_array[:, :, :3]
            
        # Ensure we have an RGB image
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img_array.shape}")
            
        # Extract color palette
        # Downsample for faster processing
        downsampled = cv2.resize(img_array, (50, 50))
        # Explicitly use shape dimensions for reshaping
        pixels = downsampled.reshape((downsampled.shape[0] * downsampled.shape[1], 3))
    except Exception as e:
        print(f"Error in generic chart extraction: {e}")
        # Return empty data as fallback
        return {
            'image_stats': {
                'width': 0,
                'height': 0,
                'aspect_ratio': 0,
                'edge_density': 0
            },
            'color_palette': [],
            'structural_elements': {
                'horizontal_lines': 0,
                'vertical_lines': 0
            },
            'chart_type': chart_type,
            'extraction_confidence': 'none',
            'note': f'Extraction failed: {str(e)}'
        }
    
    # K-means to find dominant colors
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5  # Find 5 dominant colors
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to hex colors
    palette = []
    for center in centers:
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(center[0]), int(center[1]), int(center[2])
        )
        palette.append(color_hex)
    
    # Basic image stats
    height, width = img_array.shape[:2]
    
    # Edge detection for complexity estimation
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_percentage = np.sum(edges > 0) / (height * width)
    
    # Basic structure detection
    # Check for horizontal/vertical lines (potential axes)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50, 
        minLineLength=min(width, height)//4, 
        maxLineGap=10
    )
    
    horizontal_lines = 0
    vertical_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:
                horizontal_lines += 1
            elif 80 < angle < 100:
                vertical_lines += 1
    
    return {
        'image_stats': {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'edge_density': round(edge_percentage, 4)
        },
        'color_palette': palette,
        'structural_elements': {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines
        },
        'chart_type': chart_type,
        'extraction_confidence': 'low',
        'note': 'Generic extraction used - limited accuracy'
    }
