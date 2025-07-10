import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    
    .results-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 3rem;
        border-top: 3px solid #667eea;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Placeholder function for ML prediction (replace with your actual function)
def predict_satellite_image(image):
    """
    Placeholder function for satellite image classification.
    Replace this with your actual ML model prediction function.
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: {
            'predicted_class': str,
            'confidence': float,
            'probabilities': dict
        }
    """
    # Simulate processing time
    time.sleep(1)
    
    # Placeholder class names (replace with your actual classes)
    class_names = ['Water', 'Forest', 'Urban', 'Desert', 'Agricultural', 'Cloudy']
    
    # Generate random probabilities for demonstration
    probabilities = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
    probabilities_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    predicted_class = max(probabilities_dict, key=probabilities_dict.get)
    confidence = probabilities_dict[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities_dict
    }

# Helper function to get confidence color
def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

# Main header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p>Advanced AI-powered classification of satellite imagery</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("📊 App Information")
    st.markdown("""
    **Supported Classes:**
    - 🌊 Water bodies
    - 🌲 Forest areas
    - 🏙️ Urban regions
    - 🏜️ Desert landscapes
    - 🌾 Agricultural land
    - ☁️ Cloudy areas
    
    **Supported Formats:**
    - JPG, JPEG, PNG
    - Max size: 10MB
    - Recommended: 256x256px
    """)
    
    st.header("🔍 How it works")
    st.markdown("""
    1. Upload a satellite image
    2. AI processes the image
    3. View classification results
    4. Analyze confidence scores
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="upload-section">
        <h3>📤 Upload Satellite Image</h3>
        <p>Select a satellite image to classify</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a satellite image in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Satellite Image", use_column_width=True)
        
        # Image info
        st.markdown("**Image Information:**")
        st.write(f"- **Filename:** {uploaded_file.name}")
        st.write(f"- **Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"- **Format:** {image.format}")
        st.write(f"- **Mode:** {image.mode}")

with col2:
    if uploaded_file is not None:
        st.markdown("""
        <div class="results-container">
            <h3>🔍 Classification Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Process image and get predictions
        with st.spinner("🤖 Analyzing satellite image..."):
            results = predict_satellite_image(image)
        
        # Display main prediction
        predicted_class = results['predicted_class']
        confidence = results['confidence']
        confidence_class = get_confidence_color(confidence)
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>📍 Predicted Class</h2>
            <h1>{predicted_class}</h1>
            <p class="{confidence_class}">Confidence: {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Top Prediction", predicted_class)
        with col_b:
            st.metric("Confidence", f"{confidence:.2%}")
        with col_c:
            confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
            st.metric("Reliability", confidence_level)
    
    else:
        st.info("👆 Please upload a satellite image to see classification results")

# Display probability chart if image is uploaded
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 📊 Class Probability Distribution")
    
    # Create probability chart
    probabilities = results['probabilities']
    
    # Sort probabilities for better visualization
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        # Bar chart
        fig_bar = go.Figure(data=[
            go.Bar(
                x=list(sorted_probs.keys()),
                y=list(sorted_probs.values()),
                marker_color=['#667eea' if k == predicted_class else '#94a3b8' for k in sorted_probs.keys()],
                text=[f'{v:.1%}' for v in sorted_probs.values()],
                textposition='auto',
            )
        ])
        
        fig_bar.update_layout(
            title="Classification Probabilities",
            xaxis_title="Land Cover Classes",
            yaxis_title="Probability",
            yaxis=dict(tickformat='.0%'),
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_chart2:
        # Pie chart
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=list(sorted_probs.keys()),
                values=list(sorted_probs.values()),
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3
            )
        ])
        
        fig_pie.update_layout(
            title="Distribution Overview",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed probabilities table
    st.markdown("### 📋 Detailed Classification Scores")
    
    prob_df = []
    for class_name, prob in sorted_probs.items():
        prob_df.append({
            'Class': class_name,
            'Probability': f"{prob:.4f}",
            'Percentage': f"{prob:.2%}",
            'Confidence': 'High' if prob >= 0.8 else 'Medium' if prob >= 0.6 else 'Low'
        })
    
    st.dataframe(prob_df, use_container_width=True)

# Additional analysis section
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 🔬 Advanced Analysis")
    
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        st.markdown("**🎯 Classification Summary**")
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        st.write(f"- **Prediction Entropy:** {entropy:.3f}")
        st.write(f"- **Top-2 Classes:** {list(sorted_probs.keys())[:2]}")
        st.write(f"- **Margin:** {list(sorted_probs.values())[0] - list(sorted_probs.values())[1]:.3f}")
    
    with col_analysis2:
        st.markdown("**🔍 Model Confidence Analysis**")
        if confidence >= 0.9:
            st.success("🟢 Very High Confidence - Excellent prediction quality")
        elif confidence >= 0.8:
            st.success("🟡 High Confidence - Good prediction quality")
        elif confidence >= 0.6:
            st.warning("🟠 Medium Confidence - Reasonable prediction")
        else:
            st.error("🔴 Low Confidence - Consider manual verification")

# Footer
st.markdown("""
<div class="footer">
    <h4>🛰️ Satellite Image Classifier</h4>
    <p>Powered by Advanced Machine Learning | Built with Streamlit</p>
    <p>© 2024 - Accurate land cover classification from satellite imagery</p>
    <p>
        <small>
            <strong>Note:</strong> This is a demonstration interface. 
            Replace the <code>predict_satellite_image()</code> function with your actual ML model.
        </small>
    </p>
</div>
""", unsafe_allow_html=True)

# Instructions for integration
st.markdown("---")
with st.expander("🔧 Integration Instructions"):
    st.markdown("""
    ### To integrate with your ML model:
    
    1. **Replace the `predict_satellite_image()` function** with your actual prediction function
    2. **Update class names** in the function to match your model's classes
    3. **Modify preprocessing** if needed (current assumes RGB images)
    4. **Adjust confidence thresholds** based on your model's performance
    
    ### Expected function signature:
    ```python
    def predict_satellite_image(image):
        # Your ML model code here
        return {
            'predicted_class': 'Water',  # str
            'confidence': 0.85,          # float (0-1)
            'probabilities': {           # dict
                'Water': 0.85,
                'Forest': 0.10,
                'Urban': 0.05
            }
        }
    ```
    
    ### Features included:
    - ✅ Modern, responsive UI
    - ✅ Interactive charts (Plotly)
    - ✅ Confidence visualization
    - ✅ Detailed probability analysis
    - ✅ Image information display
    - ✅ Custom CSS styling
    - ✅ Sidebar information
    - ✅ Error handling ready
    """)