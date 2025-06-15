# 🧠 Dyslexia Learning Tools Recommendation System

A sophisticated Neural Collaborative Filtering (NCF) system designed to provide personalized recommendations for dyslexia learning tools, built with PyTorch and featuring an interactive Streamlit web interface.

## 📖 Overview

This system leverages advanced deep learning techniques to analyze user demographics, learning patterns, and preferences to recommend the most suitable educational resources for individuals with dyslexia. The system combines Matrix Factorization with Multi-Layer Perceptrons in a Neural Collaborative Filtering architecture.

### 🎯 Key Features

- **Three User Types Support**: 
  - New users with stated preferences
  - New users without preferences  
  - Existing users with interaction history
- **Advanced Neural Architecture**: Hybrid NCF combining Matrix Factorization and MLPs
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Training**: Live progress tracking with performance metrics
- **Personalized Recommendations**: Category-based and collaborative filtering approaches
- **Comprehensive Analytics**: Detailed user insights and system performance metrics

### Neural Collaborative Filtering Model

The system uses a hybrid architecture that combines:
- **Matrix Factorization**: Learns latent factors for users and items
- **Multi-Layer Perceptron**: Captures complex non-linear interactions
- **Feature Integration**: Incorporates demographic and item category features

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- Minimum 4GB RAM
- 2GB free disk space

### Dependencies
torch>=2.0.0
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
optuna>=3.0.0
matplotlib>=3.5.0

## 🚀 Installation

### 1. Clone the Repository
git clone git@github.com:2-BrainCells/NCF-Recommender.git

cd dyslexia-recommendation-system

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

### 3. Install Dependencies
pip install torch streamlit pandas numpy scikit-learn optuna matplotlib

### 4. Verify Installation
python -c "import torch; print(f'PyTorch version: {torch.version}')"


### Learning Tool Categories

The system recognizes 39 different learning tools categorized into:

- **Technological Tools (T1-T17)**: Audio books, visual aids, digital tools
- **Study Strategies (S1-S22)**: Reading techniques, assessment methods, support strategies


## 🔧 Usage

### Quick Start with Streamlit App
streamlit run app.py


This launches the interactive web interface where you can:
1. **Upload Data**: Load your demographic and ratings CSV files
2. **Train Model**: Use real-time progress tracking with optimized hyperparameters
3. **Get Recommendations**: For existing users (ID 0-1204) or new users
4. **Analyze Performance**: View detailed metrics and user insights

### Command Line Interface
from recommendation_system import DyslexiaRecommendationSystem

Initialize system
rec_system = DyslexiaRecommendationSystem()

Load and preprocess data
rec_system.load_data('demographic_data.csv', 'ratings_data.csv')

Train model with optimized parameters
training_results = rec_system.train_model(epochs=20, use_best_params=True)

Get recommendations for existing user
user_profile = {'id': 400}
recommendations = rec_system.get_recommendations(user_profile, top_k=10)
print(recommendations)


### Advanced Usage with HPO
Run hyperparameter optimization
hpo_results = rec_system.run_hyperparameter_optimization(
n_trials=50,
timeout=1800
)

Train with optimized parameters
training_results = rec_system.train_model(
epochs=20,
use_best_params=True
)


## 🎮 User Types & Recommendation Logic

### 1. Existing Users (ID: 0-1204)
- **Input**: User ID only
- **Recommendation Strategy**:
  - Analyzes user's high-rated items (above personal average)
  - Recommends items from preferred categories
  - Uses collaborative filtering from similar users
  - Falls back to popular items when needed
    
user_profile = {'id': 400}
recommendations = rec_system.get_recommendations(user_profile)

### 2. New Users with Preferences
- **Input**: Demographics + category preferences
- **Recommendation Strategy**: Items from preferred categories with popularity weighting
user_profile = {
'id': 7000,
'age': 22,
'gender': 'M',
'diagnosis_timing': 'Medie',
'has_other_difficulties': 'No, solo dislessia',
'preferences': ['Digital Books', 'Online Lessons']
}

### 3. New Users without Preferences
- **Input**: Demographics only
- **Recommendation Strategy**: Popular items across all categories
user_profile = {
'id': 8000,
'age': 25,
'gender': 'F',
'diagnosis_timing': 'Superiori'
}

## 📈 Model Configuration

### Hyperparameter Search Space
- **Embedding Dimensions**: [16, 32, 64, 128, 256]
- **Hidden Layers**: 2-5 layers
- **Hidden Dimensions**: [128, 256, 512, 1024]
- **Dropout**: 0.1-0.5
- **Learning Rate**: 1e-5 to 1e-2
- **Batch Size**: [32, 64, 128, 256]


## 🎯 Performance Benchmarks

Based on testing with dyslexia learning tools dataset:

| Metric | Value |
|--------|-------|
| RMSE | 0.1847 |
| R² Score | 0.7623 |
| Precision@10 | 0.8421 |
| Recall@10 | 0.7389 |
| Training Time | ~15 minutes (20 epochs) |

## 📊 Core Components

### 1. Data Processor (`data_processor.py`)
- Handles CSV data loading and preprocessing
- Manages missing value imputation using KNN
- Creates user embeddings from demographic data
- Generates item features from category mappings

### 2. NCF Model (`model.py`)
- Hybrid Neural Collaborative Filtering architecture
- Combines Matrix Factorization and MLP branches
- Supports both fixed and HPO-suggested architectures
- Includes early stopping and weight initialization

### 3. Hyperparameter Optimizer (`hpo.py`)
- Uses Optuna for automated hyperparameter tuning
- Implements pruning and early stopping strategies
- Supports parallel optimization trials

### 4. Recommendation Engine (`recommendation_system.py`)
- Main system orchestrating all components
- Handles three different user types
- Implements personalized recommendation strategies
- Provides comprehensive user analytics

### 5. Streamlit Interface (`app.py`)
- Interactive web application
- Real-time training progress tracking
- User-friendly recommendation interface
- Visual analytics and data insights

## 🔍 Troubleshooting

### Common Issues

**CUDA out of memory:**
Reduce batch size
config['batch_size'] = 64 # or 32

**Import errors:**
Reinstall dependencies
pip install -r requirements.txt

**Data format issues:**
- Ensure CSV files have proper headers
- Check for missing values and encoding issues
- Verify user IDs are in range 0-1204 for existing users

### Performance Tips

1. **GPU Usage**: Enable CUDA for faster training
2. **Batch Size**: Adjust based on available memory
3. **Early Stopping**: Use patience=3 for optimal training time
4. **HPO**: Run optimization once, then reuse best parameters
