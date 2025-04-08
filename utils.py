import numpy as np
import cv2
from PIL import Image
import io
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def get_sample_charts(sample_name):
    """
    Generate sample visualization images for demonstration.
    
    Args:
        sample_name (str): Name of the sample chart to generate
        
    Returns:
        io.BytesIO: Image file in BytesIO format
    """
    # Set up the figure
    plt.figure(figsize=(8, 6), dpi=100)
    
    # Generate different sample visualizations based on the name
    if sample_name == "Bar Chart with Too Many Categories":
        # Create a bar chart with too many categories (exceeds optimal data density)
        categories = [f"Category {i}" for i in range(1, 21)]
        values = np.random.randint(10, 100, size=20)
        
        # Use a 3D effect to further complicate the visualization
        ax = plt.axes(projection='3d')
        ax.bar3d(
            np.arange(len(categories)), 
            np.zeros(len(categories)), 
            np.zeros(len(categories)), 
            0.8, 0.8, values, 
            shade=True
        )
        
        plt.xticks(np.arange(len(categories)), categories, rotation=90)
        plt.xlabel("Categories")
        plt.ylabel("Values")
        plt.title("Sales by Product Category (3D)")
        plt.tight_layout()
        
    elif sample_name == "Overly Complex Pie Chart":
        # Create a pie chart with too many slices (exceeds cognitive threshold)
        labels = [f"Segment {i}" for i in range(1, 15)]  # Increased to 14 segments
        
        # Create some very small segments that will be hard to see
        sizes = [30, 20, 15, 10, 7, 5, 3, 2, 2, 1.5, 1.5, 1, 1, 0.5]
        
        # Use explode to complicate the visual even more
        explode = tuple([0.1 if i % 3 == 0 else 0 for i in range(len(labels))])
        
        # Use the least readable text color with low contrast
        plt.pie(
            sizes, labels=labels, autopct='%1.1f%%', 
            startangle=90, explode=explode,
            textprops={'color': 'darkgray'}
        )
        
        # Add shadow effect to further complicate the visual
        plt.axis('equal')
        plt.title("Market Share Distribution with Many Small Segments")
        
    elif sample_name == "Line Chart with Poor Color Choices":
        # Create a line chart with poor color choices and too many lines
        x = np.linspace(0, 10, 100)
        
        # Create multiple lines with similar colors (exceeds working memory capacity)
        plt.plot(x, np.sin(x), color='#FF0000', label='Series 1', linewidth=1)
        plt.plot(x, np.sin(x+0.5), color='#FF0040', label='Series 2', linewidth=1)
        plt.plot(x, np.sin(x+1), color='#FF0080', label='Series 3', linewidth=1)
        plt.plot(x, np.sin(x+1.5), color='#FF00C0', label='Series 4', linewidth=1)
        plt.plot(x, np.sin(x+2), color='#FF00FF', label='Series 5', linewidth=1)
        plt.plot(x, np.sin(x+2.5), color='#C000FF', label='Series 6', linewidth=1)
        plt.plot(x, np.sin(x+3), color='#8000FF', label='Series 7', linewidth=1)
        plt.plot(x, np.sin(x+3.5), color='#4000FF', label='Series 8', linewidth=1)
        
        # Use a low contrast grid
        plt.grid(True, color='#E0E0E0', linestyle='-', linewidth=0.5)
        
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Trend Analysis with Multiple Series")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
    elif sample_name == "Scatter Plot with Insufficient Labels":
        # Create a scatter plot with insufficient labeling and overplotting
        np.random.seed(0)
        
        # Create clusters of points that will overlap
        x1 = np.random.normal(30, 5, 100)
        y1 = np.random.normal(30, 5, 100)
        
        x2 = np.random.normal(70, 10, 200)
        y2 = np.random.normal(70, 10, 200)
        
        # Create highly skewed outlier points
        x3 = np.random.uniform(0, 10, 20)
        y3 = np.random.uniform(80, 100, 20)
        
        # Combine all points
        x = np.concatenate([x1, x2, x3])
        y = np.concatenate([y1, y2, y3])
        
        # Plot without alpha transparency, creating overplotting
        plt.scatter(x, y, color='blue', s=20)
        
        # Minimal labels
        plt.title("Data Distribution")
        
        # No axes labels or grid - poor readability
        plt.grid(False)
        
    else:
        # Default empty plot if no valid sample is selected
        plt.text(0.5, 0.5, "No sample selected", 
                 horizontalalignment='center', verticalalignment='center')
        
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Close the figure to avoid memory leaks
    plt.close()
    
    return buf

def save_processed_file(image, chart_type=None, filename="processed_visualization.png"):
    """
    Save a processed visualization file.
    
    Args:
        image: The image to save
        chart_type (str, optional): Type of chart for specialized processing
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    # Save to BytesIO
    buf = io.BytesIO()
    
    if isinstance(image, Image.Image):
        image.save(buf, format='PNG')
    else:
        # Convert numpy array to PIL Image and save
        Image.fromarray(img_array).save(buf, format='PNG')
        
    buf.seek(0)
    return buf

def extract_text_from_image(image):
    """
    Extract potential text elements from an image.
    This is a placeholder for more advanced OCR integration.
    
    Args:
        image (PIL.Image): The input image
        
    Returns:
        dict: Detected text elements with positions
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use MSER (Maximally Stable Extremal Regions) to detect text regions
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    # Process regions to identify potential text elements
    text_regions = []
    for i, region in enumerate(regions):
        # Convert region to bounding box
        x, y, w, h = cv2.boundingRect(region)
        
        # Filter out regions that are too small or too large
        if h > 10 and w > 5 and h/w < 10 and w/h < 10:
            text_regions.append({
                'id': i,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'text': f"Text {i}"  # Placeholder for actual OCR
            })
    
    return {
        'regions': text_regions,
        'count': len(text_regions)
    }

def create_color_palette(colors, size=(300, 50)):
    """
    Create a color palette image from a list of colors.
    
    Args:
        colors (list): List of color strings in hex format (#RRGGBB)
        size (tuple): Size of the output image
        
    Returns:
        PIL.Image: Color palette image
    """
    # Create an empty image
    palette = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    if not colors:
        return Image.fromarray(palette)
    
    # Calculate the width of each color segment
    segment_width = size[0] // len(colors)
    
    # Fill the palette with colors
    for i, color in enumerate(colors):
        # Parse hex color
        try:
            hex_color = color.lstrip('#')
            r, g, b = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
            
            # Fill segment
            start_x = i * segment_width
            end_x = (i + 1) * segment_width if i < len(colors) - 1 else size[0]
            palette[:, start_x:end_x] = [r, g, b]
        except Exception:
            # Use gray for invalid colors
            palette[:, i*segment_width:(i+1)*segment_width] = [128, 128, 128]
    
    return Image.fromarray(palette)

def get_chart_improvement_examples():
    """
    Generate before/after examples of chart improvements.
    This is a placeholder for more advanced visualization generation.
    
    Returns:
        dict: Examples with before/after images and descriptions
    """
    # In a real application, this would return actual improvement examples
    # This is just a placeholder structure
    return {
        'bar_chart': {
            'title': 'Bar Chart Improvements',
            'description': 'Sorting bars by value, adding clear labels, and using a consistent color scheme.',
            'before_image': None,  # Would be an actual image in a real implementation
            'after_image': None
        },
        'pie_chart': {
            'title': 'Pie Chart Improvements',
            'description': 'Reducing the number of segments, adding percentage labels, and using distinctive colors.',
            'before_image': None,
            'after_image': None
        },
        'line_chart': {
            'title': 'Line Chart Improvements',
            'description': 'Reducing visual noise, highlighting important trends, and improving color contrast.',
            'before_image': None,
            'after_image': None
        }
    }
