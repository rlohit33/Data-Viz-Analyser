# Import the chart suggestion function
from suggest_alternative_chart_new import suggest_alternative_chart

def generate_recommendations(analysis_results, chart_type):
    """
    Generate actionable recommendations to improve the visualization based on analysis results.
    
    Args:
        analysis_results (dict): Results from chart analysis
        chart_type (str): Type of the chart ('bar', 'line', 'pie', 'scatter', etc.)
        
    Returns:
        list: List of recommendation dictionaries with title, issue, and recommendation
    """
    recommendations = []
    
    # Calculate estimated data points for more specific recommendations
    data_point_count = estimate_data_points(analysis_results, chart_type)
    analysis_results['data_points'] = data_point_count
    
    # Categorize and group issues for prioritized recommendations
    data_structure_issues = []
    aesthetic_issues = []
    readability_issues = []
    
    # Categorize issues by importance
    for issue in analysis_results.get('issues', []):
        # Classify issues by type
        if any(term in issue.lower() for term in ['many bars', 'many segments', 'data points', 'density']):
            data_structure_issues.append(issue)
        elif any(term in issue.lower() for term in ['color', 'contrast', 'vibrant', 'saturation']):
            aesthetic_issues.append(issue)
        else:
            readability_issues.append(issue)
    
    # First add recommendations for data structure issues (highest priority)
    for issue in data_structure_issues:
        recommendation = create_recommendation_from_issue(issue, chart_type)
        if recommendation:
            recommendation['example'] = enhance_with_quantitative_insights(
                recommendation.get('example', ''), 
                issue, 
                chart_type,
                data_point_count
            )
            recommendations.append(recommendation)
    
    # Then add recommendations for readability issues (medium priority)
    for issue in readability_issues:
        recommendation = create_recommendation_from_issue(issue, chart_type)
        if recommendation:
            recommendation['example'] = enhance_with_quantitative_insights(
                recommendation.get('example', ''), 
                issue, 
                chart_type,
                data_point_count
            )
            recommendations.append(recommendation)
    
    # Finally, consolidate all aesthetic issues into a single recommendation (lowest priority)
    if aesthetic_issues:
        consolidated_rec = {
            'title': 'Improve Visual Aesthetics',
            'issue': 'Multiple visual aesthetic issues detected',
            'recommendation': 'Consider enhancing the overall visual appeal with these improvements:',
            'example': '• ' + '\n• '.join(aesthetic_issues)
        }
        recommendations.append(consolidated_rec)
    
    # Add only the most important chart-specific recommendation if there's room
    scores = analysis_results.get('scores', {})
    lowest_score_category = None
    lowest_score = 10  # Start with maximum score
    
    # Find the category with the lowest score
    for category, score in scores.items():
        if score < lowest_score:
            lowest_score = score
            lowest_score_category = category
    
    # Only add a recommendation for the most problematic category if score is low
    if lowest_score_category and lowest_score < 5:
        if lowest_score_category == 'color':
            rec = get_color_recommendations(chart_type)[0] if get_color_recommendations(chart_type) else None
        elif lowest_score_category == 'composition':
            rec = get_composition_recommendations(chart_type)[0] if get_composition_recommendations(chart_type) else None
        elif lowest_score_category == 'readability':
            rec = get_readability_recommendations(chart_type)[0] if get_readability_recommendations(chart_type) else None
        elif lowest_score_category == 'chart_specific':
            rec = get_specific_chart_recommendations(chart_type)[0] if get_specific_chart_recommendations(chart_type) else None
        else:
            rec = None
            
        if rec:
            rec['example'] = enhance_with_quantitative_insights(
                rec.get('example', ''),
                f"{lowest_score_category} issues",
                chart_type,
                data_point_count
            )
            recommendations.append(rec)
    
    # Consider alternative chart types suggestion
    alt_chart_recommendation = suggest_alternative_chart(analysis_results, chart_type)
    if alt_chart_recommendation:
        recommendations.append(alt_chart_recommendation)
    
    # Ensure we return at least some recommendations
    if not recommendations:
        general_recs = get_general_recommendations(chart_type)
        for rec in general_recs:
            rec['example'] = enhance_with_quantitative_insights(
                rec.get('example', ''),
                'general improvement',
                chart_type,
                data_point_count
            )
        recommendations.extend(general_recs)
    
    # Cap at 5 most important recommendations
    return recommendations[:5]


def estimate_data_points(analysis_results, chart_type):
    """
    Estimate the number of data points in the visualization based on analysis results
    
    Args:
        analysis_results (dict): Results from chart analysis
        chart_type (str): Type of the chart
        
    Returns:
        int: Estimated number of data points
    """
    issues = analysis_results.get('issues', [])
    
    # Try to extract numeric data point counts from issues
    for issue in issues:
        # Look for numbers followed by data-related terms
        if any(term in issue.lower() for term in ['point', 'bar', 'line', 'slice', 'segment']):
            words = issue.split()
            for i, word in enumerate(words):
                if word.isdigit() and i > 0:
                    return int(word)
    
    # Default estimates based on chart type if no specific count found
    if chart_type == 'pie':
        return 7  # Average pie chart segments
    elif chart_type == 'bar':
        return 12  # Average bar chart categories
    elif chart_type == 'line':
        return 20  # Average line chart data points
    elif chart_type == 'scatter':
        return 100  # Average scatter plot points
    else:
        return 15  # Generic estimate for other charts
        

def enhance_with_quantitative_insights(example_text, issue_type, chart_type, data_point_count):
    """
    Add quantitative research insights to recommendation examples
    
    Args:
        example_text (str): Original example text
        issue_type (str): The type of issue being addressed
        chart_type (str): Type of the chart
        data_point_count (int): Estimated number of data points
        
    Returns:
        str: Enhanced example with quantitative insights
    """
    # If example already contains quantitative information, return as is
    if any(term in example_text.lower() for term in ['%', 'ratio', 'threshold', 'pixel']):
        return example_text
    
    insights = {
        'pie': {
            'color issues': "Research shows color discrimination beyond 7±2 hues reduces accuracy by 24%.",
            'readability issues': "Labels should occupy at least 3% of chart area to maintain 92% readability.",
            'composition issues': f"With {data_point_count} segments, your pie chart exceeds the optimal 5-7 segment threshold where angle comparison becomes 30% less accurate.",
            'chart specific issues': "Pie charts with >7 segments reduce value comparison accuracy by up to 35% compared to bar charts.",
            'general improvement': "Using sorted segments (largest to smallest) improves comparison accuracy by 15%."
        },
        'bar': {
            'color issues': "Using a maximum of 7 distinct colors improves categorical recognition by 24%.",
            'readability issues': "Text labels should maintain min 9pt (12px) size for >98% legibility across devices.",
            'composition issues': f"Optimal bar width-to-spacing ratio is 2:1 for maximum discriminability. With {data_point_count} bars, consider aggregation.",
            'chart specific issues': "Horizontally oriented bars can accommodate 50% more categories while maintaining readability.",
            'general improvement': "Sorting bars by value rather than alphabetically improves analysis speed by 20-30%."
        },
        'line': {
            'color issues': "Line thickness should be >1.5px and use a minimum color contrast ratio of 3:1 against the background.",
            'readability issues': "Adding data points markers at regular intervals improves value estimation accuracy by 18%.",
            'composition issues': f"With {data_point_count} data points, line charts approach optimal density at 0.025 points per pixel.",
            'chart specific issues': "Limit to 4-5 lines per chart to stay within working memory capacity (Miller's Law).",
            'general improvement': "Subtle grid lines improve value estimation accuracy by 12% with minimal visual noise."
        },
        'scatter': {
            'color issues': "Using color encoding for a third variable increases information density by 40% without reducing comprehension.",
            'readability issues': "Point sizes should scale with √n to maintain visual prominence as n increases.",
            'composition issues': f"With {data_point_count} points, optimal opacity is {min(100, max(10, int(500/data_point_count**0.5)))}% to balance visibility and overplotting.",
            'chart specific issues': "Adding trend lines improves pattern recognition by 35% with large point collections.",
            'general improvement': "For n>500 points, binning techniques (hexbin) improve pattern recognition by 45%."
        }
    }
    
    # Handle unknown chart types
    if chart_type not in insights:
        return example_text + " Following visualization best practices improves comprehension speed by 30% and accuracy by 15-40%."
    
    # Handle unknown issue types
    if issue_type.lower() not in insights[chart_type]:
        issue_type = 'general improvement'  # Default to general improvement
    
    # Return enhanced example
    return example_text + " " + insights[chart_type][issue_type.lower()]

def create_recommendation_from_issue(issue, chart_type):
    """Convert an analysis issue into a structured recommendation"""
    # Common issues and their recommendations
    recommendation_map = {
        # Color-related issues
        "Limited color variety detected": {
            "title": "Enhance Color Differentiation",
            "issue": "Limited color variety makes it difficult to distinguish between data points/categories.",
            "recommendation": "Use a more diverse color palette with clearly distinguishable colors for different data categories.",
            "example": "For categorical data, consider using ColorBrewer palettes or Tableau's color schemes."
        },
        "Too many colors may overwhelm": {
            "title": "Simplify Color Palette",
            "issue": "Using too many colors can create visual noise and confusion.",
            "recommendation": "Reduce the number of colors by grouping related categories or using sequential/gradient colors for related data.",
            "example": "Group minor categories into 'Other' and reserve distinct colors for important categories only."
        },
        "Colors appear to be low in saturation": {
            "title": "Increase Color Vibrancy",
            "issue": "Low color saturation reduces visual impact and can make the chart appear faded.",
            "recommendation": "Increase color saturation for main data elements while keeping background and supporting elements more neutral.",
            "example": "Use saturated colors (60-80% saturation) for data elements and light neutral colors for background elements."
        },
        "Low contrast between elements": {
            "title": "Improve Color Contrast",
            "issue": "Poor contrast makes it difficult to distinguish between different elements in your visualization.",
            "recommendation": "Increase the contrast between foreground and background elements, and between different data series.",
            "example": "Ensure a contrast ratio of at least 3:1 between data elements and 4.5:1 for any text elements."
        },
        "Red-green color combination detected": {
            "title": "Use Colorblind-Friendly Palette",
            "issue": "Red-green color combinations are problematic for people with color vision deficiencies.",
            "recommendation": "Replace red-green combinations with blue-orange or purple-green palettes that are more accessible.",
            "example": "Use the Viridis, Magma, or Cividis color palettes which are designed to be colorblind-friendly."
        },
        
        # Composition issues
        "Chart appears to be visually complex": {
            "title": "Simplify Visual Elements",
            "issue": "Excessive visual elements create cognitive overload and distract from the data.",
            "recommendation": "Remove decorative elements, minimize grid lines, and focus on the essential data components.",
            "example": "Apply the data-ink ratio principle: maximize the ink used for data representation, minimize non-data ink."
        },
        "Few or no grid lines detected": {
            "title": "Add Appropriate Grid Lines",
            "issue": "Lack of reference lines makes it difficult to read precise values from the visualization.",
            "recommendation": "Add subtle grid lines to help viewers trace values accurately across the chart.",
            "example": "Use light gray horizontal grid lines that don't compete with the data elements."
        },
        "Pie chart has many segments": {
            "title": "Reduce Pie Chart Segments",
            "issue": "Too many segments in a pie chart make it difficult to compare values and perceive proportions.",
            "recommendation": "Limit pie charts to 5-7 segments maximum; group smaller categories into an 'Other' category.",
            "example": "Keep only the top 5 categories as individual slices and group the rest into 'Other'."
        },
        "Chart appears crowded": {
            "title": "Increase Whitespace",
            "issue": "Insufficient spacing between elements creates visual clutter and reduces readability.",
            "recommendation": "Add more margin space around the chart and increase spacing between elements within the chart.",
            "example": "Ensure at least 10-15% margin around the entire visualization and appropriate spacing between bars/lines."
        },
        
        # Readability issues
        "Few or no text elements detected": {
            "title": "Improve Labeling",
            "issue": "Insufficient labeling makes the visualization difficult to interpret without context.",
            "recommendation": "Add a clear title, axis labels, and data labels where appropriate to make the chart self-explanatory.",
            "example": "Include a concise title that explains what the data shows, label both axes, and consider direct data labeling for important values."
        },
        "Low text contrast detected": {
            "title": "Enhance Text Visibility",
            "issue": "Poor contrast between text and background reduces readability.",
            "recommendation": "Increase the contrast between text and its background; use dark text on light backgrounds or vice versa.",
            "example": "Use black or dark gray text on white/light backgrounds with a minimum contrast ratio of 4.5:1."
        },
        "Image appears to be blurry": {
            "title": "Improve Image Clarity",
            "issue": "Blurry or low-resolution elements reduce the professional appearance and readability.",
            "recommendation": "Use higher resolution images and vector-based elements where possible.",
            "example": "Export visualizations at 2x the intended display size or use SVG format for web display."
        },
        
        # Bar chart specific
        "Too many bars detected": {
            "title": "Reduce Number of Bars",
            "issue": "Too many bars create visual clutter and make comparison difficult.",
            "recommendation": "Focus on the most important categories or group minor categories; consider using a horizontal bar chart for many categories.",
            "example": "Show only the top 10 categories and group others, or split the data into multiple charts by a logical grouping."
        },
        "Possible 3D effect detected": {
            "title": "Remove 3D Effects",
            "issue": "3D effects distort data perception and can lead to misinterpretation of values.",
            "recommendation": "Use flat 2D bars which allow for more accurate visual comparison of values.",
            "example": "Convert all 3D visual elements to simple 2D representations focused on the data."
        },
        
        # Line chart specific
        "Many lines detected": {
            "title": "Reduce Number of Lines",
            "issue": "Too many lines create visual clutter and make it difficult to follow individual trends.",
            "recommendation": "Highlight only the most important 3-5 trends or split into multiple charts.",
            "example": "Create separate charts for different categories or highlight the most important line and make others more subtle."
        },
        "Many line intersections detected": {
            "title": "Minimize Line Crossings",
            "issue": "Numerous intersecting lines create confusion and make trends difficult to follow.",
            "recommendation": "Consider small multiples, different chart types, or highlighting specific periods/trends.",
            "example": "Create separate charts for different time periods or use visual cues (thickness, color) to emphasize key lines."
        },
        
        # Pie chart specific
        "No clear circular shape detected": {
            "title": "Improve Pie Chart Structure",
            "issue": "The pie chart shape is not clearly defined which can distort proportion perception.",
            "recommendation": "Ensure the pie chart has a clear circular shape without distortion or tilting.",
            "example": "Use a simple 2D pie chart with clear segments and avoid perspective or 3D effects."
        },
        "Chart contains very small slices": {
            "title": "Eliminate Tiny Slices",
            "issue": "Very small slices are difficult to perceive and label appropriately.",
            "recommendation": "Combine small slices into an 'Other' category that represents no more than 10-15% of the total.",
            "example": "Group all categories representing less than 5% of the total into a single 'Other' category."
        },
        
        # Scatter plot specific
        "Large number of scatter points": {
            "title": "Manage Point Density",
            "issue": "Too many data points create overplotting and obscure patterns.",
            "recommendation": "Use transparency, adjust point size, or consider density plots for large datasets.",
            "example": "Set point opacity to 20-30% to reveal overlapping points, or use a hexbin plot for very large datasets."
        },
        "Axis lines not clearly detected": {
            "title": "Improve Axis Definition",
            "issue": "Poorly defined axes make it difficult to interpret the positions of data points.",
            "recommendation": "Add clear x and y axes with appropriate tick marks and labels.",
            "example": "Include solid axis lines with regularly spaced tick marks and clear numeric labels."
        },
        "Data points appear clustered": {
            "title": "Address Data Clustering",
            "issue": "Clustered data points make it difficult to see individual values and patterns.",
            "recommendation": "Consider logarithmic scales, zooming in on relevant regions, or using interactive features.",
            "example": "Apply a log transformation to spread out clustered points or create multiple views focusing on different regions."
        }
    }
    
    # Check if we have a direct match for the issue
    for key, rec in recommendation_map.items():
        if key in issue:
            return rec
    
    # No direct match, create a generic recommendation
    words = issue.split()
    if len(words) > 3:
        title_words = [w.capitalize() for w in words[:3] if len(w) > 3]
        if title_words:
            title = " ".join(title_words)
            return {
                "title": title,
                "issue": issue,
                "recommendation": "Address this issue to improve your visualization's effectiveness."
            }
    
    return None

def get_color_recommendations(chart_type):
    """Generate color-specific recommendations based on chart type"""
    recommendations = []
    
    # General color recommendations
    general_rec = {
        "title": "Use an Effective Color Scheme",
        "issue": "Suboptimal color choices can hinder data interpretation and accessibility.",
        "recommendation": "Apply a purposeful color scheme appropriate to your data type.",
        "example": "For sequential data, use single-hue progressions; for categorical data, use distinct colors with similar lightness."
    }
    
    # Chart-specific color recommendations
    if chart_type == 'bar':
        recommendations.append({
            "title": "Optimize Bar Chart Colors",
            "issue": "Ineffective color application in bar charts can distract from data comparison.",
            "recommendation": "Use a single color for simple comparisons or a logical color scheme for categories.",
            "example": "For a single metric across categories, use the same color with different saturation levels to indicate value differences."
        })
    elif chart_type == 'line':
        recommendations.append({
            "title": "Improve Line Chart Color Distinction",
            "issue": "Poor color differentiation makes it difficult to distinguish between multiple lines.",
            "recommendation": "Use clearly distinguishable colors for different lines and ensure sufficient contrast with the background.",
            "example": "Limit to 4-6 distinct line colors and consider using both color and line patterns (solid, dashed) for better differentiation."
        })
    elif chart_type == 'pie':
        recommendations.append({
            "title": "Enhance Pie Chart Color Segmentation",
            "issue": "Inadequate color distinction between pie segments reduces comprehension.",
            "recommendation": "Use contrasting colors for adjacent segments and ensure logical color progression if data is sequential.",
            "example": "For categorical data, use distinct but harmonious colors; for ranked data, use a sequential color scheme from light to dark."
        })
    
    recommendations.append(general_rec)
    return recommendations[:2]  # Return at most 2 recommendations

def get_composition_recommendations(chart_type):
    """Generate composition-specific recommendations based on chart type"""
    recommendations = []
    
    # General composition recommendation
    general_rec = {
        "title": "Improve Visual Hierarchy",
        "issue": "Lack of clear visual hierarchy makes it difficult to focus on key information.",
        "recommendation": "Establish a clear visual hierarchy by emphasizing important elements and de-emphasizing secondary elements.",
        "example": "Use size, color, and position to highlight key data points, make supporting elements (axes, labels) more subtle."
    }
    
    # Chart-specific composition recommendations
    if chart_type == 'bar':
        recommendations.append({
            "title": "Optimize Bar Chart Layout",
            "issue": "Ineffective spacing or orientation reduces bar chart readability.",
            "recommendation": "Adjust bar spacing, consider orientation (vertical vs horizontal), and sort data meaningfully.",
            "example": "For many categories, use horizontal bars sorted by value; maintain consistent spacing between bars (50-80% of bar width)."
        })
    elif chart_type == 'line':
        recommendations.append({
            "title": "Enhance Line Chart Clarity",
            "issue": "Cluttered line chart reduces the ability to follow trends.",
            "recommendation": "Simplify the chart by removing unnecessary grid lines, using appropriate line thickness, and adding subtle reference lines.",
            "example": "Use line thickness of 2-3px for primary lines, add light horizontal grid lines at major axis intervals only."
        })
    elif chart_type == 'pie':
        recommendations.append({
            "title": "Improve Pie Chart Proportions",
            "issue": "Poor segment arrangement in pie charts impacts interpretation of proportions.",
            "recommendation": "Start the largest segment at 12 o'clock position and proceed clockwise with decreasing sizes.",
            "example": "Arrange slices in size order starting from the top, or in a logical sequence if the data has natural ordering."
        })
    elif chart_type == 'scatter':
        recommendations.append({
            "title": "Optimize Scatter Plot Layout",
            "issue": "Ineffective use of the plotting area reduces pattern visibility.",
            "recommendation": "Adjust axis scales to maximize use of the plot area and consider adding reference lines for important values.",
            "example": "Set axis ranges to slightly exceed data extremes and add reference lines for averages or thresholds."
        })
    
    recommendations.append(general_rec)
    return recommendations[:2]  # Return at most 2 recommendations

def get_readability_recommendations(chart_type):
    """Generate readability-specific recommendations based on chart type"""
    recommendations = []
    
    # General readability recommendation
    general_rec = {
        "title": "Enhance Overall Readability",
        "issue": "Poor readability elements reduce the visualization's effectiveness.",
        "recommendation": "Improve text elements with clear hierarchy, appropriate font sizes, and proper contrast.",
        "example": "Use a consistent font family, with sizes of at least 12pt for labels, 14pt for axis titles, and 16-18pt for chart titles."
    }
    
    # Chart-specific readability recommendations
    if chart_type == 'bar':
        recommendations.append({
            "title": "Improve Bar Chart Labeling",
            "issue": "Ineffective labeling in bar charts reduces data interpretation accuracy.",
            "recommendation": "Add clear value labels for bars and ensure category labels are legible and unambiguous.",
            "example": "Position value labels inside bars (for longer bars) or outside (for shorter bars), rotate category labels if needed."
        })
    elif chart_type == 'line':
        recommendations.append({
            "title": "Enhance Line Chart Annotations",
            "issue": "Lack of proper annotations makes trend interpretation difficult.",
            "recommendation": "Add clear data markers at key points and annotate significant changes or events.",
            "example": "Use data points at regular intervals, highlight extremes or turning points, and add callouts for notable events."
        })
    elif chart_type == 'pie':
        recommendations.append({
            "title": "Improve Pie Chart Labeling",
            "issue": "Poor labeling of pie segments reduces comprehension of proportions.",
            "recommendation": "Include percentage or value labels directly on segments or use a clean legend.",
            "example": "Add percentage labels directly on larger segments and use a pull-out with labels for smaller segments."
        })
    elif chart_type == 'scatter':
        recommendations.append({
            "title": "Enhance Scatter Plot Interpretation",
            "issue": "Lack of reference points makes it difficult to interpret scatter plot patterns.",
            "recommendation": "Add trend lines, highlight significant points, and ensure axes have clear scales and labels.",
            "example": "Add a regression line to show overall trend, highlight outliers, and include grid lines for value reference."
        })
    
    recommendations.append(general_rec)
    return recommendations[:2]  # Return at most 2 recommendations

def get_specific_chart_recommendations(chart_type):
    """Generate chart-type specific recommendations"""
    if chart_type == 'bar':
        return [{
            "title": "Optimize Bar Chart Design",
            "issue": "Bar charts can be optimized for better data comparison.",
            "recommendation": "Sort bars by value (not alphabetically) unless there's a natural order, and consider using horizontal orientation for many categories.",
            "example": "For 8+ categories, use horizontal bars sorted from highest to lowest value for easier comparison."
        }]
    elif chart_type == 'line':
        return [{
            "title": "Improve Line Chart Effectiveness",
            "issue": "Line charts need clear trend visibility and appropriate scaling.",
            "recommendation": "Ensure y-axis scaling appropriately represents the data range and consider adding smoothing for noisy data.",
            "example": "Start y-axis at zero for volume metrics, but use a narrower range for metrics showing relative change. Add smoothing for high-frequency data."
        }]
    elif chart_type == 'pie':
        return [{
            "title": "Consider Alternatives to Pie Chart",
            "issue": "Pie charts are often not the most effective way to show proportional data.",
            "recommendation": "For precise comparison of values, consider using a bar chart instead of a pie chart.",
            "example": "Horizontal bar charts sorted by value provide more accurate comparison of proportions than pie charts."
        }]
    elif chart_type == 'scatter':
        return [{
            "title": "Enhance Scatter Plot Information Density",
            "issue": "Scatter plots may not convey enough information about underlying patterns.",
            "recommendation": "Add trend lines, confidence intervals, or density indicators to reveal patterns.",
            "example": "Include a regression line with confidence bands, or use color/size to encode additional variables."
        }]
    else:
        return [{
            "title": "Optimize Chart Selection",
            "issue": "The chosen chart type may not be optimal for your data.",
            "recommendation": "Ensure your chart type matches your intended message and data structure.",
            "example": "Use bar charts for comparing categories, line charts for trends over time, scatter plots for correlations, and pie charts sparingly for parts of a whole (with few categories)."
        }]

# The suggest_alternative_chart function has been moved to a separate file: suggest_alternative_chart_new.py

def get_general_recommendations(chart_type):
    """Provide general improvement recommendations when specific issues aren't detected"""
    recommendations = []
    
    # General recommendation for all chart types
    recommendations.append({
        "title": "Apply Data Visualization Best Practices",
        "issue": "General improvements can enhance any visualization's effectiveness.",
        "recommendation": "Focus on clarity, remove chart junk, and ensure the visualization tells a clear story.",
        "example": "Remove decorative elements, ensure title clearly states the main insight, and make sure the most important data stands out visually."
    })
    
    # Add a chart-specific general recommendation
    if chart_type == 'bar':
        recommendations.append({
            "title": "Optimize Bar Chart Fundamentals",
            "issue": "Basic bar chart improvements can enhance readability.",
            "recommendation": "Sort bars meaningfully, use appropriate spacing, and ensure clear labeling.",
            "example": "Sort bars by value unless a different order is more meaningful, use consistent spacing between bars, and add direct value labels."
        })
    elif chart_type == 'line':
        recommendations.append({
            "title": "Enhance Line Chart Clarity",
            "issue": "Basic line chart improvements can make trends more apparent.",
            "recommendation": "Use appropriate line thickness, add data markers at key points, and consider annotations for important events.",
            "example": "Use 2-3px lines, add markers at regular intervals or key data points, and annotate significant trend changes."
        })
    elif chart_type == 'pie':
        recommendations.append({
            "title": "Maximize Pie Chart Effectiveness",
            "issue": "Basic pie chart improvements can enhance proportion perception.",
            "recommendation": "Limit to 5-7 segments, start largest segment at 12 o'clock, and add clear percentage labels.",
            "example": "Group small segments into 'Other', order segments by size (unless another order is meaningful), and add direct percentage labels."
        })
    elif chart_type == 'scatter':
        recommendations.append({
            "title": "Improve Scatter Plot Interpretability",
            "issue": "Basic scatter plot improvements can reveal patterns more clearly.",
            "recommendation": "Add trend lines, adjust point opacity for overlapping areas, and ensure axis scales maximize data visibility.",
            "example": "Add a regression line to show overall trend, use 20-30% opacity for points, and adjust axis scales to focus on the main data cluster."
        })
    
    return recommendations[:3]  # Return up to 3 general recommendations
