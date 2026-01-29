"""
Solar Panel Condition Classification App
Author: Alexander Olomukoro
Description: AI-powered solar panel defect detection using EfficientNetB0
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# Import configuration
from config import (
    MODEL_PATH, IMG_SIZE, CLASSES, CLASS_DESCRIPTIONS,
    SEVERITY_LEVELS, RECOMMENDED_ACTIONS, SEVERITY_COLORS,
    APP_TITLE, APP_ICON, PAGE_LAYOUT, AUTHOR, VERSION,
    MODEL_METRICS
)

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.02);
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        text-align: center;
    }
    .info-box {
        padding: 1rem;
        border-left: 4px solid #667eea;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-clean {
        color: #10b981;
        font-weight: bold;
    }
    .status-defect {
        color: #ef4444;
        font-weight: bold;
    }
    h1 {
        color: #1f2937;
    }
    h2 {
        color: #374151;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #4b5563;
    }
    </style>
    """, unsafe_allow_html=True)

# Configuration is imported from config.py

# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def load_model():
    """Load the trained model with caching."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file is in the 'models/' directory.")
        return None

def preprocess_image(image):
    """Preprocess image for EfficientNet."""
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    return img_array

def create_probability_chart(predictions, classes):
    """Create an interactive bar chart for predictions."""
    df = pd.DataFrame({
        'Condition': classes,
        'Probability': predictions[0] * 100
    }).sort_values('Probability', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Condition'],
        orientation='h',
        marker=dict(
            color=df['Probability'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Confidence %")
        ),
        text=[f"{p:.1f}%" for p in df['Probability']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Prediction Confidence for All Classes",
        xaxis_title="Confidence (%)",
        yaxis_title="Panel Condition",
        height=400,
        template="plotly_white",
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

def create_gauge_chart(confidence):
    """Create a gauge chart for confidence level."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Prediction Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, template="plotly_white")
    return fig

def save_prediction_history(image_name, prediction, confidence):
    """Save prediction to history file."""
    history_file = 'data/prediction_history.json'
    
    os.makedirs('data', exist_ok=True)
    
    record = {
        'timestamp': datetime.now().isoformat(),
        'image': image_name,
        'prediction': prediction,
        'confidence': float(confidence)
    }
    
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(record)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save prediction history: {str(e)}")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/solar-panel.png", width=80)
    st.title("☀️ Solar Panel AI Inspector")
    st.markdown("---")
    
    # Updated Statistics Section
    st.markdown("### 📈 Model Performance")
    
    # Updated Statistics Section
    st.markdown("### 📈 Model Performance")
    
    acc = MODEL_METRICS['Accuracy']
    insight = MODEL_METRICS['Key_Insight']

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", acc)
    with col_b:
        st.metric("Top F1", "0.95")
    
    st.info(f"💡 **Insight:** {insight}")

    with st.expander("Detailed F1-Scores"):
        scores = MODEL_METRICS['F1_Scores']
        for k, v in scores.items():
            st.caption(f"**{k}:** {v}")

    st.markdown("---")
    
    st.markdown("""
    ### About This App
    
    This AI-powered application uses **deep learning** to detect defects 
    and conditions in solar panels.
    
    **Model:** EfficientNetB0 (Fine-Tuned)  
    **Classes:** 6 conditions
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### How to Use
    
    1. Upload a solar panel image
    2. AI analyzes the image
    3. Get instant results with recommendations
    4. View detailed confidence scores
    
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Class Information
    """)
    
    for cls in CLASSES:
        with st.expander(cls):
            st.write(f"**Description:** {CLASS_DESCRIPTIONS[cls]}")
            st.write(f"**Severity:** {SEVERITY_LEVELS[cls]}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <small>Built with ❤️ by Alexander Olomukoro<br>
    Powered by TensorFlow & Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================
# Header
st.title("☀️ Solar Panel Condition Classification")
st.markdown("""
<div class='info-box'>
<b>AI-Powered Inspection System</b><br>
Upload an image of your solar panel to instantly detect defects, contamination, 
or other conditions that may affect performance.
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🔄 Loading AI model..."):
    model = load_model()

if model is None:
    st.stop()

# File uploader
st.markdown("## 📤 Upload Solar Panel Image")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of your solar panel for analysis"
    )

with col2:
    st.markdown("""
    **Tips for best results:**
    - Use clear, well-lit images
    - Capture the entire panel
    - Avoid blurry photos
    - Multiple angles help
    """)

# Process uploaded image
if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    st.markdown("---")
    st.markdown("## 📸 Uploaded Image")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Your Solar Panel Image", use_container_width=True)
    
    # Analyze button
    st.markdown("---")
    
    if st.button("🔍 Analyze Panel Condition", type="primary"):
        # Preprocess
        img_array = preprocess_image(image)
        
        # Predict with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 AI is analyzing your image...")
        progress_bar.progress(30)
        
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = CLASSES[predicted_idx]
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Save to history
        save_prediction_history(uploaded_file.name, predicted_class, confidence)
        
        # Results section
        st.markdown("---")
        st.markdown("## 🎯 Analysis Results")
        
        # Main prediction card
        severity = SEVERITY_LEVELS[predicted_class]
        severity_color = SEVERITY_COLORS[severity]
        
        st.markdown(f"""
        <div style='padding: 2rem; border-radius: 15px; 
                    background: linear-gradient(135deg, {severity_color}20 0%, {severity_color}40 100%);
                    border: 3px solid {severity_color}; text-align: center;'>
            <h1 style='color: {severity_color}; margin: 0; font-size: 2.5rem;'>
                {predicted_class}
            </h1>
            <p style='font-size: 1.3rem; margin: 0.5rem 0; color: #1f2937;'>
                <b>Confidence:</b> {confidence:.1%}
            </p>
            <p style='font-size: 1rem; margin: 0; color: #4b5563;'>
                <b>Severity Level:</b> {severity}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Prediction</h3>
                <p style='font-size: 1.5rem; font-weight: bold; color: {severity_color};'>
                    {predicted_class}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Confidence</h3>
                <p style='font-size: 1.5rem; font-weight: bold; color: #667eea;'>
                    {confidence:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Severity</h3>
                <p style='font-size: 1.5rem; font-weight: bold; color: {severity_color};'>
                    {severity}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed information tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Confidence Analysis", "📋 Recommendations", 
                                           "📈 All Predictions", "ℹ️ Details"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(confidence), 
                              use_container_width=True)
            
            with col2:
                # Top 3 predictions
                st.markdown("### Top 3 Predictions")
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                
                for i, idx in enumerate(top_indices):
                    class_name = CLASSES[idx]
                    prob = predictions[0][idx]
                    
                    if i == 0:
                        st.markdown(f"🥇 **{class_name}** - {prob:.1%}")
                    elif i == 1:
                        st.markdown(f"🥈 {class_name} - {prob:.1%}")
                    else:
                        st.markdown(f"🥉 {class_name} - {prob:.1%}")
                
                # Confidence interpretation
                st.markdown("---")
                st.markdown("### Confidence Interpretation")
                if confidence >= 0.9:
                    st.success("✅ Very High Confidence - Highly reliable prediction")
                elif confidence >= 0.75:
                    st.info("ℹ️ High Confidence - Reliable prediction")
                elif confidence >= 0.5:
                    st.warning("⚠️ Moderate Confidence - Consider manual inspection")
                else:
                    st.error("❌ Low Confidence - Manual inspection strongly recommended")
        
        with tab2:
            st.markdown("### 💡 Recommended Actions")
            st.info(RECOMMENDED_ACTIONS[predicted_class])
            
            st.markdown("### 📝 Detailed Analysis")
            st.write(f"**Condition:** {predicted_class}")
            st.write(f"**Description:** {CLASS_DESCRIPTIONS[predicted_class]}")
            st.write(f"**Severity Level:** {severity}")
            
            if severity in ['High', 'Medium']:
                st.error("⚠️ **Important:** This condition may significantly impact panel performance. "
                        "Consider taking immediate action.")
            elif severity == 'Low':
                st.warning("⚡ **Note:** While not critical, addressing this condition can improve efficiency.")
            else:
                st.success("✅ **Good News:** Your solar panel is in optimal condition!")
        
        with tab3:
            st.markdown("### 📊 All Class Probabilities")
            
            # Interactive chart
            st.plotly_chart(create_probability_chart(predictions, CLASSES), 
                          use_container_width=True)
            
            # Data table
            st.markdown("### 📋 Detailed Breakdown")
            prob_df = pd.DataFrame({
                'Condition': CLASSES,
                'Confidence': [f"{p:.2%}" for p in predictions[0]],
                'Probability': predictions[0]
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(
                prob_df[['Condition', 'Confidence']].reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
        
        with tab4:
            st.markdown("### ℹ️ Technical Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Model Information**
                - Architecture: EfficientNetB0
                - Transfer Learning: ImageNet
                - Input Size: 224x224
                - Framework: TensorFlow/Keras
                """)
            
            with col2:
                st.markdown("""
                **Image Information**
                - File Name: {}
                - Size: {} x {}
                - Format: {}
                - Analysis Time: {}
                """.format(
                    uploaded_file.name,
                    image.size[0],
                    image.size[1],
                    image.format,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            
            st.markdown("---")
            st.markdown("### 🎯 Classification Metrics")
            st.write("- Number of Classes: 6")
            st.write(f"- Model Accuracy: {MODEL_METRICS['Accuracy']}")
            st.write("- Prediction Method: Softmax Classification")

else:
    # Instructions when no image uploaded
    st.markdown("---")
    st.info("👆 Please upload a solar panel image to begin analysis")
    
    # Sample instructions with visual
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What to Expect
        
        Once you upload an image, our AI will:
        
        1. **Analyze** the panel condition
        2. **Classify** into 6 categories
        3. **Provide** confidence scores
        4. **Recommend** actions
        5. **Display** detailed insights
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Supported Conditions
        
        - ✅ Clean panels
        - 🦅 Bird droppings
        - 💨 Dust accumulation
        - ⚡ Electrical damage
        - 🔨 Physical damage
        - ❄️ Snow coverage
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>
        <b>Solar Panel AI Inspector</b> | Powered by Deep Learning<br>
        Built with TensorFlow, Keras, and Streamlit<br>
        © 2026 Alexander Olomukoro. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)
