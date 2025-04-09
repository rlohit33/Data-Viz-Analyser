import streamlit as st
import os
import numpy as np
import json
import tempfile
from PIL import Image
import io

# Custom modules
from chart_detector import detect_chart_type
from chart_analyzer import analyze_chart
from visualization_recommender import generate_recommendations
from suggest_alternative_chart_new import suggest_alternative_chart
from json_converter import visualization_to_json
from utils import get_sample_charts, save_processed_file

st.set_page_config(
    page_title="Data Visualization Analyzer",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Data Visualization Analyzer & Recommender")
    
    st.markdown("""
    This tool helps you improve your data visualizations through data-driven analysis.
    Upload your chart or visualization image to:
    - Convert it to structured JSON data
    - Analyze its effectiveness with quantitative metrics
    - Get analytical recommendations with specific thresholds and research-backed insights
    """)
    
    # Sidebar for upload and options
    with st.sidebar:
        st.header("Upload & Options")
        
        uploaded_file = st.file_uploader(
            "Upload a visualization image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"]
        )
        
        st.divider()
        
        st.subheader("Examples")
        st.markdown("Don't have a visualization? Try one of our samples:")
        
        sample_options = [
            "None",
            "Bar Chart with Too Many Categories",
            "Overly Complex Pie Chart",
            "Line Chart with Poor Color Choices",
            "Scatter Plot with Insufficient Labels"
        ]
        
        sample_choice = st.selectbox("Sample Visualizations", sample_options)
        
        if sample_choice != "None":
            sample_file = get_sample_charts(sample_choice)
            if uploaded_file is None and sample_file is not None:
                uploaded_file = sample_file
    
    # Main content
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        try:
            # Process the uploaded file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = Image.open(io.BytesIO(file_bytes))
            
            # Print image information for debugging
            st.write(f"Image format: {image.format}, mode: {image.mode}, size: {image.size}")
            
            # Ensure we're working with RGB image format (convert if needed)
            if image.mode == 'RGBA':
                st.info("Converting image from RGBA to RGB format")
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                st.info(f"Converting image from {image.mode} to RGB format")
                image = image.convert('RGB')
            
            # Display the original visualization
            with col1:
                st.subheader("Original Visualization")
                st.image(image, use_container_width=True)
            
            # Process the visualization
            with st.spinner("Analyzing your visualization..."):
                # Detect the chart type
                chart_type = detect_chart_type(image)
                print(f"DEBUG - App: Detected chart type from chart_detector.py: {chart_type}")
                
                # Convert visualization to JSON
                json_data = visualization_to_json(image, chart_type)
                
                # Analyze the chart
                analysis_results = analyze_chart(image, chart_type)
                print(f"DEBUG - App: Analysis results issues: {analysis_results.get('issues', [])}")
                
                # Generate recommendations (now includes alternative chart suggestions)
                recommendations = generate_recommendations(analysis_results, chart_type)
                
                # Debug print recommendations for testing
                print(f"DEBUG - App: Generated {len(recommendations)} recommendations")
                
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            st.info("Please try uploading a different image file.")
            return
        
        # Display analysis results
        with col2:
            st.subheader("Analysis & Recommendations")
            
            st.write(f"**Detected Chart Type:** {chart_type}")
            
            st.markdown("### Extracted Data (JSON)")
            with st.expander("View JSON Data"):
                st.json(json_data)
            
            st.markdown("### Visualization Score")
            score = analysis_results.get('overall_score', 0)
            st.progress(score / 10)
            st.write(f"Overall Score: {score}/10")
            
            st.markdown("### Recommendations")
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['title']}"):
                    st.markdown(f"**Issue:** {rec['issue']}")
                    st.markdown(f"**Recommendation:** {rec['recommendation']}")
                    if 'example' in rec:
                        st.markdown(f"**Example:** {rec['example']}")
        
        # Export options
        st.divider()
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Export Options")
            
            export_json_btn = st.download_button(
                label="Export JSON Data",
                data=json.dumps(json_data, indent=2),
                file_name="visualization_data.json",
                mime="application/json"
            )
            
            export_recommendations = st.download_button(
                label="Export Recommendations (JSON)",
                data=json.dumps(recommendations, indent=2),
                file_name="visualization_recommendations.json",
                mime="application/json"
            )
    
    else:
        # Display instructions when no file is uploaded
        st.info("👈 Please upload a visualization image or select a sample from the sidebar to get started")
        
        # Information about supported chart types
        st.subheader("Supported Chart Types")
        st.markdown("""
        This tool can analyze the following chart types:
        - Bar charts (vertical and horizontal)
        - Line charts
        - Pie charts
        - Scatter plots
        - Area charts
        - Bubble charts
        
        For best results, ensure your visualization has clear elements and is not overly complex.
        """)
        
        # Information about the analysis process
        st.subheader("How It Works")
        st.markdown("""
        1. **Upload** your visualization image
        2. Our system **detects** the chart type using computer vision techniques
        3. Image processing extracts the **visual elements** and estimates data point density
        4. The visualization is **converted to JSON** data for further analysis
        5. We **quantitatively analyze** your visualization against research-backed thresholds
        6. You receive **analytical recommendations** with specific thresholds and metrics
        """)

if __name__ == "__main__":
    main()
