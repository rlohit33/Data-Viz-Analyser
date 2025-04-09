# Data Visualization Analyzer & Recommender

A powerful tool that helps you improve your data visualizations through data-driven analysis. This application provides automated analysis and recommendations for your charts and visualizations.

## Features

- **Chart Type Detection**: Automatically identifies the type of chart in your visualization
- **JSON Conversion**: Converts visualizations to structured JSON data
- **Quantitative Analysis**: Analyzes visualization effectiveness using research-backed metrics
- **Smart Recommendations**: Provides specific recommendations for improvement
- **Sample Visualizations**: Includes example charts for testing and learning

## Supported Chart Types

- Bar charts (vertical and horizontal)
- Line charts
- Pie charts
- Scatter plots
- Area charts
- Bubble charts

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd Data-Viz-Analyser
```

2. Install the required dependencies:
```bash
pip install -r project_requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
python -m streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:5000)

3. Upload a visualization image or select a sample from the sidebar

4. View the analysis results and recommendations

## Project Structure

- `app.py`: Main Streamlit application
- `chart_detector.py`: Chart type detection module
- `chart_analyzer.py`: Visualization analysis module
- `visualization_recommender.py`: Recommendation generation module
- `json_converter.py`: Visualization to JSON conversion module
- `utils.py`: Utility functions
- `suggest_alternative_chart_new.py`: Alternative chart suggestion module

## Dependencies

- matplotlib >= 3.5.0
- numpy >= 1.20.0
- opencv-python >= 4.5.0
- Pillow >= 9.0.0
- scikit-image >= 0.19.0
- streamlit >= 1.18.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Contact

[Your contact information] 