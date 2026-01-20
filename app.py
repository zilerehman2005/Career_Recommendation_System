import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.graph_objects as go
from career_data import get_career_roadmap, get_job_search_links, EDUCATION_LEVELS, SKILLS_LIST, INTERESTS_LIST

# Set page config
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .roadmap-step {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        border-radius: 5px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .job-link {
        background: #3498db;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s;
    }
    .job-link:hover {
        background: #2980b9;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Load models and metadata
@st.cache_resource
def load_models():
    try:
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        models = []
        for i in range(5):
            model = MLP(metadata['num_features'], metadata['num_classes'])
            model.load_state_dict(torch.load(f'models/model_{i}.pth', map_location=torch.device('cpu')))
            model.eval()
            models.append(model)
        
        return models, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_input(age, education, skills, interests, metadata):
    """Preprocess user input to match training data format"""
    # Create base dataframe
    input_data = pd.DataFrame({
        'Age': [age]
    })
    
    # Add education one-hot encoding
    for edu in metadata['education_columns']:
        edu_name = edu.replace('Edu_', '')
        input_data[edu] = 1 if edu_name == education else 0
    
    # Add skills one-hot encoding
    for skill_col in metadata['skill_columns']:
        skill_name = skill_col.replace('Skills_', '')
        input_data[skill_col] = 1 if skill_name in skills else 0
    
    # Add interests one-hot encoding
    for interest_col in metadata['interest_columns']:
        interest_name = interest_col.replace('Interests_', '')
        input_data[interest_col] = 1 if interest_name in interests else 0
    
    # Scale age
    input_data['Age'] = metadata['scaler'].transform([[age]])[0][0]
    
    # Ensure column order matches training data
    input_data = input_data[metadata['feature_names']]
    
    return torch.tensor(input_data.values, dtype=torch.float32)

def ensemble_predict(models, input_tensor):
    """Get ensemble prediction with confidence scores"""
    probs = []
    with torch.no_grad():
        for model in models:
            out = torch.softmax(model(input_tensor), dim=1)
            probs.append(out.cpu().numpy())
    
    # Average probabilities across models
    ensemble_prob = np.mean(probs, axis=0)[0]
    predicted_idx = np.argmax(ensemble_prob)
    confidence = ensemble_prob[predicted_idx]
    
    return predicted_idx, confidence, ensemble_prob

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 24}},
        delta = {'reference': 80, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccb'},
                {'range': [50, 75], 'color': '#fff4cc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_top_predictions_chart(career_names, probabilities):
    """Create bar chart for top 5 career predictions"""
    # Get top 5 predictions
    top_5_idx = np.argsort(probabilities)[-5:][::-1]
    top_careers = [career_names[i] for i in top_5_idx]
    top_probs = [probabilities[i] * 100 for i in top_5_idx]
    
    fig = go.Figure([go.Bar(
        x=top_probs,
        y=top_careers,
        orientation='h',
        marker=dict(
            color=top_probs,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence %")
        ),
        text=[f'{p:.1f}%' for p in top_probs],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Top 5 Career Recommendations",
        xaxis_title="Confidence (%)",
        yaxis_title="Career",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig

# Main App
def main():
    st.markdown("<h1>🎯 AI Career Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #7f8c8d;'>Discover your ideal career path powered by ensemble deep learning</p>", unsafe_allow_html=True)
    
    # Load models
    models, metadata = load_models()
    
    if models is None:
        st.error(" Models not found. Please train the model first by running 'train_model.py'")
        return
    
    # Sidebar for input
    with st.sidebar:
        st.header(" Your Information")
        st.markdown("---")
        
        # Age input
        age = st.slider("Age", min_value=18, max_value=65, value=25, help="Select your current age")
        
        # Education input
        education = st.selectbox(
            "Education Level",
            options=EDUCATION_LEVELS,
            help="Select your highest level of education"
        )
        
        # Skills input (multi-select)
        skills = st.multiselect(
            "Skills",
            options=SKILLS_LIST,
            help="Select all skills you possess (minimum 1, maximum 5 recommended)"
        )
        
        # Interests input (multi-select)
        interests = st.multiselect(
            "Interests",
            options=INTERESTS_LIST,
            help="Select your professional interests (minimum 1, maximum 3 recommended)"
        )
        
        st.markdown("---")
        predict_button = st.button(" Get Career Recommendation", use_container_width=True)
    
    # Main content area
    if predict_button:
        # Validation
        if not skills:
            st.warning(" Please select at least one skill")
            return
        if not interests:
            st.warning(" Please select at least one interest")
            return
        
        with st.spinner(" Analyzing your profile with ensemble AI models..."):
            # Preprocess input
            input_tensor = preprocess_input(age, education, skills, interests, metadata)
            
            # Get prediction
            predicted_idx, confidence, all_probs = ensemble_predict(models, input_tensor)
            predicted_career = metadata['career_classes'][predicted_idx]
            
            # Display results
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="margin: 0; font-size: 2.5rem;">🎉 Recommended Career</h2>
                    <h1 style="margin: 1rem 0; font-size: 3rem;">{predicted_career}</h1>
                    <p style="font-size: 1.2rem; opacity: 0.9;">Based on your skills, interests, and background</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_top_predictions_chart(metadata['career_classes'], all_probs), use_container_width=True)
            
            with col3:
                st.markdown("###  Your Profile Summary")
                st.markdown(f"**Age:** {age}")
                st.markdown(f"**Education:** {education}")
                st.markdown(f"**Skills:** {len(skills)} selected")
                st.markdown(f"**Interests:** {len(interests)} selected")
                st.markdown("---")
                st.markdown(f"**Model Confidence:** {confidence*100:.2f}%")
                st.markdown(f"**Prediction Quality:** {'Excellent' if confidence > 0.8 else 'Good' if confidence > 0.6 else 'Fair'}")
            
            # Career Roadmap Section
            st.markdown("---")
            st.markdown("##  Your Career Roadmap")
            
            roadmap = get_career_roadmap(predicted_career)
            
            if roadmap:
                for i, step in enumerate(roadmap, 1):
                    st.markdown(f"""
                        <div class="roadmap-step">
                            <h3>Step {i}: {step['title']}</h3>
                            <p>{step['description']}</p>
                            <p><strong>Duration:</strong> {step['duration']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Roadmap data is being prepared for this career path.")
            
            # Job Search Links Section
            st.markdown("---")
            st.markdown("# Find Jobs")
            
            job_links = get_job_search_links(predicted_career)
            
            cols = st.columns(3)
            for i, (platform, url) in enumerate(job_links.items()):
                with cols[i % 3]:
                    st.markdown(f'<a href="{url}" target="_blank" class="job-link">{platform}</a>', unsafe_allow_html=True)
            
            # Alternative career suggestions
            st.markdown("---")
            st.markdown("# Alternative Career Paths")
            
            # Get top 3 alternatives (excluding the top prediction)
            top_5_idx = np.argsort(all_probs)[-5:][::-1]
            alternatives = []
            for idx in top_5_idx[1:4]:  # Skip first (main prediction), take next 3
                alternatives.append({
                    'career': metadata['career_classes'][idx],
                    'confidence': all_probs[idx] * 100
                })
            
            alt_cols = st.columns(3)
            for i, alt in enumerate(alternatives):
                with alt_cols[i]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #3498db; margin-top: 0;">{alt['career']}</h3>
                            <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 0;">{alt['confidence']:.1f}%</p>
                            <p style="color: #7f8c8d; margin: 0;">Match Score</p>
                        </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
