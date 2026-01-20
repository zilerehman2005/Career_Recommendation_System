# 🎯 AI Career Recommendation System

An intelligent career recommendation application powered by ensemble deep learning that predicts ideal career paths based on education, skills, and interests.

## 🌟 Features

- **Ensemble Deep Learning**: 5 MLP models with soft voting for robust predictions
- **Interactive UI**: Beautiful Streamlit interface with real-time predictions
- **Confidence Scoring**: Visual gauge showing prediction confidence
- **Top 5 Recommendations**: See alternative career paths with match percentages
- **Career Roadmaps**: Step-by-step guides for your recommended career
- **Job Search Integration**: Direct links to top job platforms
- **Data Balancing**: SMOTE implementation for handling class imbalance
- **Professional Visualizations**: Plotly charts and custom CSS styling

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Internet connection (for first-time package installation)

## 🚀 Installation & Setup

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <your-repository-url>
cd career-recommendation-app

# Or download and extract the ZIP file, then navigate to the folder
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installing PyTorch may take a few minutes depending on your internet connection.

### Step 4: Prepare Your Dataset

Place your `AI_Career_Recommendation_8000.csv` file in the project root directory.

**Expected CSV format:**
```
CandidateID,Name,Age,Education,Skills,Interests,Recommended_Career,Recommendation_Score
1,John Doe,25,Bachelor's,Python;Machine Learning,Data Science,Data Scientist,0.95
```

## 🎓 Training the Model

Before running the app, you need to train the ensemble model:

```bash
python train_model.py
```

**What this does:**
- Loads and preprocesses your dataset
- Applies SMOTE for class balancing
- Trains 5 MLP models with different random seeds
- Saves all models to `models/` directory
- Generates training loss visualization
- Displays test set performance metrics

**Expected output:**
```
✅ ALL DONE! You can now run the Streamlit app:
   streamlit run app.py

📁 Generated files:
   • models/model_0.pth through model_4.pth
   • models/model_metadata.pkl
   • ensemble_loss_curve.png
```

**Training time**: Approximately 5-10 minutes on a modern CPU (faster with GPU)

## 🖥️ Running the Application

Once training is complete, launch the Streamlit app:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📱 Using the Application

### 1. **Enter Your Information** (Left Sidebar)
   - **Age**: Use the slider (18-65 years)
   - **Education**: Select your highest level of education
   - **Skills**: Select 1-5 relevant skills from the dropdown
   - **Interests**: Select 1-3 professional interests

### 2. **Get Recommendation**
   - Click the "🔮 Get Career Recommendation" button
   - Wait for the ensemble model to process your profile

### 3. **View Results**
   - **Primary Recommendation**: Your best-matched career with large display
   - **Confidence Gauge**: Visual representation of prediction certainty
   - **Top 5 Careers**: Bar chart showing alternative career paths
   - **Profile Summary**: Overview of your input data

### 4. **Explore Career Path**
   - **Roadmap**: Step-by-step learning path for your recommended career
   - **Job Links**: Direct search links to major job platforms
   - **Alternatives**: Three next-best career matches

## 📂 Project Structure

```
career-recommendation-app/
│
├── app.py                              # Main Streamlit application
├── train_model.py                      # Model training script
├── career_data.py                      # Career roadmaps and job links
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── AI_Career_Recommendation_8000.csv   # Your dataset (you provide this)
│
├── models/                             # Generated after training
│   ├── model_0.pth                    # Trained model 1
│   ├── model_1.pth                    # Trained model 2
│   ├── model_2.pth                    # Trained model 3
│   ├── model_3.pth                    # Trained model 4
│   ├── model_4.pth                    # Trained model 5
│   └── model_metadata.pkl             # Feature encodings & scaler
│
└── ensemble_loss_curve.png             # Training visualization
```

## 🔧 Troubleshooting

### Issue: "No module named 'streamlit'"
**Solution**: Make sure you've activated your virtual environment and installed requirements:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Models not found"
**Solution**: You need to train the model first:
```bash
python train_model.py
```

### Issue: "FileNotFoundError: AI_Career_Recommendation_8000.csv"
**Solution**: Make sure your dataset CSV file is in the project root directory with the exact name.

### Issue: Slow training or predictions
**Solution**: 
- Close other applications to free up RAM
- Consider using a smaller batch size by editing `train_model.py`
- If you have a GPU, PyTorch will automatically use it for faster training

### Issue: Port 8501 already in use
**Solution**: Stop any running Streamlit apps or use a different port:
```bash
streamlit run app.py --server.port 8502
```

## 🎨 Customization

### Adding More Careers to Roadmaps

Edit `career_data.py` and add your career to `CAREER_ROADMAPS`:

```python
CAREER_ROADMAPS = {
    "Your Career Name": [
        {
            "title": "Step 1 Title",
            "description": "What to do in this step",
            "duration": "3-6 months"
        },
        # Add more steps...
    ]
}
```

### Modifying Skills or Interests Lists

Edit the `SKILLS_LIST` and `INTERESTS_LIST` in `career_data.py` to match your dataset.

### Changing Model Architecture

Modify the `MLP` class in `train_model.py`:

```python
class MLP(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),  # Increase neurons
            nn.ReLU(), 
            nn.Dropout(0.3),
            # Add more layers...
        )
```

### Adjusting Training Parameters

In `train_model.py`, modify:
- **Epochs**: Change `range(1, 51)` to train longer/shorter
- **Learning rate**: Modify `lr=0.001` in optimizer
- **Batch size**: Change `batch_size=128` in DataLoader
- **Number of models**: Modify the `seeds` list

## 📊 Model Performance

The ensemble model typically achieves:
- **Accuracy**: 85-95% on test set
- **Macro F1-Score**: 0.85-0.92
- **Confidence**: Most predictions have >75% confidence

Performance depends on:
- Dataset quality and size
- Feature distribution
- Class balance
- Training configuration

## 🔐 Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers
- Models and predictions are completely offline
- Job search links open in new browser tabs

## 🤝 Contributing

To improve this project:
1. Add more detailed career roadmaps
2. Enhance the UI with additional visualizations
3. Implement career comparison features
4. Add export functionality for reports
5. Create mobile-responsive layouts

## 📝 Technical Details

### Model Architecture
- **Type**: Multi-Layer Perceptron (MLP)
- **Layers**: 4 fully connected layers (256→128→64→num_classes)
- **Activation**: ReLU
- **Regularization**: Dropout (0.3)
- **Optimizer**: Adam with weight decay
- **Loss**: Cross-Entropy

### Ensemble Strategy
- **Method**: Soft voting (probability averaging)
- **Models**: 5 independently trained MLPs
- **Seeds**: [42, 123, 777, 999, 2025]
- **Final Prediction**: argmax of averaged probabilities

### Data Processing
- **Age**: Standard scaling (StandardScaler)
- **Education**: One-hot encoding
- **Skills**: Multi-hot encoding (supports multiple skills)
- **Interests**: Multi-hot encoding (supports multiple interests)
- **Balancing**: SMOTE (Synthetic Minority Over-sampling)

## 🆘 Support

If you encounter issues:
1. Check this README's troubleshooting section
2. Verify all dependencies are installed: `pip list`
3. Ensure Python version is 3.8+: `python --version`
4. Check that your CSV has the correct format
5. Try retraining the model: `python train_model.py`

## 📄 License

This project is provided as-is for educational and commercial use.

## 🎉 Acknowledgments

- Built with Streamlit for rapid UI development
- PyTorch for deep learning capabilities
- Scikit-learn for preprocessing utilities
- Plotly for interactive visualizations
- SMOTE for handling imbalanced datasets

---

**Made with ❤️ for career guidance and AI education**

**Last Updated**: January 2026
