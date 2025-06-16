import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
import torch
import time
warnings.filterwarnings('ignore')

from recommendation_system import DyslexiaRecommendationSystem
from config import CATEGORY_MAPPING, DEFAULT_CONFIG

st.set_page_config(
    page_title="🧠 Dyslexia Learning Tools Recommender",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .rec-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-card {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'rec_system' not in st.session_state:
    st.session_state.rec_system = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def initialize_system():
    """Initialize the dyslexia recommendation system instance and manage session state for persistent system access across Streamlit reruns."""
    if st.session_state.rec_system is None:
        st.session_state.rec_system = DyslexiaRecommendationSystem()
    return st.session_state.rec_system

def train_model_with_real_progress(rec_system, epochs=20):
    """Execute model training with real-time progress visualization and comprehensive metrics display in Streamlit interface."""
    st.subheader("🎯 Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        loss_metric = st.empty()
    with col2:
        val_metric = st.empty()
    with col3:
        epoch_metric = st.empty()
    with col4:
        time_metric = st.empty()
    
    details_container = st.expander("📊 Training Details", expanded=False)
    
    try:
        start_time = time.time()
        
        rec_system.embedding_dims = DEFAULT_CONFIG['embedding_dims']
        rec_system.hidden_dims = DEFAULT_CONFIG['hidden_dims']
        rec_system.dropout = DEFAULT_CONFIG['dropout']
        rec_system.learning_rate = DEFAULT_CONFIG['learning_rate']
        rec_system.weight_decay = DEFAULT_CONFIG['weight_decay']
        rec_system.batch_size = DEFAULT_CONFIG['batch_size']
        
        status_text.text("🔧 Preparing data...")
        progress_bar.progress(0.05)
        
        train_data, val_data, test_data = rec_system.data_processor.split_data(
            rec_system.data_processor.user_data,
            rec_system.data_processor.item_data,
            rec_system.data_processor.ratings_data,
            0.2
        )
        
        status_text.text("🔧 Initializing model...")
        progress_bar.progress(0.1)
        
        num_users = rec_system.data_processor.num_users
        num_items = rec_system.data_processor.num_items
        user_feature_dim = rec_system.data_processor.user_data.shape[1]
        item_feature_dim = rec_system.data_processor.item_data.shape[1]
        
        from model import NeuralCollaborativeFiltering, TrainingEarlyStopping
        
        rec_system.model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim,
            embedding_dims=rec_system.embedding_dims,
            hidden_dims=rec_system.hidden_dims,
            dropout=rec_system.dropout,
            device=rec_system.device
        )
        
        rec_system.model.to(rec_system.device)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            rec_system.model.parameters(),
            lr=rec_system.learning_rate,
            weight_decay=rec_system.weight_decay
        )
        
        early_stopping = TrainingEarlyStopping(patience=3, delta=0.0002)
        
        from torch.utils.data import DataLoader, TensorDataset
        
        X_user_train, X_item_train, y_train = train_data
        X_user_val, X_item_val, y_val = val_data
        
        train_dataset = TensorDataset(
            torch.IntTensor(y_train[:, 0]),
            torch.IntTensor(y_train[:, 1]),
            torch.FloatTensor(X_user_train),
            torch.FloatTensor(X_item_train),
            torch.FloatTensor(y_train[:, 2])
        )
        
        val_dataset = TensorDataset(
            torch.IntTensor(y_val[:, 0]),
            torch.IntTensor(y_val[:, 1]),
            torch.FloatTensor(X_user_val),
            torch.FloatTensor(X_item_val),
            torch.FloatTensor(y_val[:, 2])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=rec_system.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=rec_system.batch_size, shuffle=False)
        
        history = {'loss': [], 'val_loss': []}
        
        status_text.text("🚀 Starting training...")
        progress_bar.progress(0.15)
        
        for epoch in range(epochs):
            epoch_progress = 0.15 + (epoch / epochs) * 0.75
            progress_bar.progress(epoch_progress)
            status_text.text(f"Training Epoch {epoch + 1}/{epochs}")
            
            rec_system.model.train()
            total_train_loss = 0.0
            batch_count = 0
            
            for batch_idx, (user_ids, item_ids, user_features, item_features, ratings) in enumerate(train_loader):
                user_ids = user_ids.to(rec_system.device)
                item_ids = item_ids.to(rec_system.device)
                user_features = user_features.to(rec_system.device)
                item_features = item_features.to(rec_system.device)
                ratings = ratings.to(rec_system.device)
                
                optimizer.zero_grad()
                predictions = rec_system.model(user_ids, item_ids, user_features, item_features)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                batch_count += 1
                
                batch_progress = (batch_idx + 1) / len(train_loader)
                current_epoch_progress = epoch + batch_progress
                overall_progress = 0.15 + (current_epoch_progress / epochs) * 0.75
                progress_bar.progress(min(overall_progress, 0.9))
            
            avg_train_loss = total_train_loss / batch_count
            history['loss'].append(avg_train_loss)
            
            rec_system.model.eval()
            total_val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for user_ids, item_ids, user_features, item_features, ratings in val_loader:
                    user_ids = user_ids.to(rec_system.device)
                    item_ids = item_ids.to(rec_system.device)
                    user_features = user_features.to(rec_system.device)
                    item_features = item_features.to(rec_system.device)
                    ratings = ratings.to(rec_system.device)
                    
                    predictions = rec_system.model(user_ids, item_ids, user_features, item_features)
                    loss = criterion(predictions, ratings)
                    total_val_loss += loss.item()
                    val_batch_count += 1
            
            avg_val_loss = total_val_loss / val_batch_count
            history['val_loss'].append(avg_val_loss)
            
            loss_metric.metric("Training Loss", f"{avg_train_loss:.4f}")
            val_metric.metric("Validation Loss", f"{avg_val_loss:.4f}")
            epoch_metric.metric("Epoch", f"{epoch + 1}/{epochs}")
            
            elapsed_time = time.time() - start_time
            time_metric.metric("Elapsed Time", f"{elapsed_time:.0f}s")
            
            with details_container:
                st.text(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            early_stopping(avg_val_loss, rec_system.model)
            if early_stopping.early_stop:
                st.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            time.sleep(0.1)
        
        status_text.text("📊 Evaluating model...")
        progress_bar.progress(0.95)
        
        from utils import evaluate_model
        test_metrics = evaluate_model(rec_system.model, test_data, rec_system.batch_size, criterion, rec_system.device)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training Completed Successfully!")
        
        return {
            'training_history': history,
            'test_metrics': test_metrics
        }
        
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Training failed!")
        return None

def get_user_info(rec_system, user_id):
    """Retrieve comprehensive user analytics including rating patterns, preferences, and behavioral insights for profile analysis."""
    try:
        user_ratings = rec_system.data_processor.ratings_data[
            rec_system.data_processor.ratings_data['user_id'] == user_id
        ]
        
        if len(user_ratings) == 0:
            return None
        
        stats = {
            'total_ratings': len(user_ratings),
            'avg_rating': round(user_ratings['rating'].mean() * 5, 2),
            'min_rating': round(user_ratings['rating'].min() * 5, 2),
            'max_rating': round(user_ratings['rating'].max() * 5, 2)
        }
        
        top_items = []
        for _, row in user_ratings.nlargest(5, 'rating').iterrows():
            item_code = rec_system.data_processor.item_id_to_code.get(row['item_id'], f"ITEM_{row['item_id']}")
            category = CATEGORY_MAPPING.get(item_code, 'Other')
            top_items.append({
                'item_code': item_code,
                'category': category,
                'rating': round(row['rating'] * 5, 2)
            })
        
        category_preferences = {}
        for category in set(CATEGORY_MAPPING.values()):
            category_items = []
            for item_id, item_code in rec_system.data_processor.item_id_to_code.items():
                if CATEGORY_MAPPING.get(item_code, 'Other') == category:
                    category_items.append(item_id)
            
            if category_items:
                category_ratings = user_ratings[user_ratings['item_id'].isin(category_items)]
                if len(category_ratings) > 0:
                    avg_rating = round(category_ratings['rating'].mean() * 5, 2)
                    category_preferences[category] = {
                        'average_rating': avg_rating,
                        'items_rated': len(category_ratings)
                    }
        
        sorted_categories = sorted(
            category_preferences.items(), 
            key=lambda x: x[1]['average_rating'], 
            reverse=True
        )
        
        return {
            'stats': stats, 
            'top_items': top_items,
            'category_preferences': dict(sorted_categories[:5]),
            'preference_quality': 'high' if stats['avg_rating'] >= 3.5 else 'moderate' if stats['avg_rating'] >= 2.5 else 'low'
        }
        
    except Exception as e:
        st.error(f"Error getting user info: {str(e)}")
        return None

def main():
    """Main Streamlit application interface providing comprehensive dyslexia learning tool recommendation system with training, analytics, and user interaction capabilities."""
    st.markdown('<h1 class="main-header">🧠 Dyslexia Learning Tools Recommender</h1>', unsafe_allow_html=True)
    st.markdown("Complete Neural Collaborative Filtering System for Dyslexia Learning Tool Recommendations")
    
    rec_system = initialize_system()
    
    tab1, tab2, tab3 = st.tabs(["🔧 System Setup", "👤 Get Recommendations", "📊 Analytics"])
    
    with tab1:
        st.header("🔧 System Setup & Training")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.data_loaded:
                st.success("✅ Data Loaded")
            else:
                st.error("❌ No Data")
        with col2:
            if st.session_state.model_trained:
                st.success("✅ Model Trained")
            else:
                st.error("❌ No Model")
        with col3:
            try:
                if st.session_state.data_loaded:
                    st.info(f"📊 {rec_system.data_processor.num_users} users, {rec_system.data_processor.num_items} items")
                else:
                    st.info("📊 No data statistics")
            except:
                st.info("📊 No data statistics")
        
        st.markdown("---")
        
        st.subheader("📊 Step 1: Load Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            demographic_file = st.file_uploader(
                "Upload Demographic CSV",
                type=['csv'],
                help="Upload your demographic data file"
            )
        
        with col2:
            ratings_file = st.file_uploader(
                "Upload Ratings CSV",
                type=['csv'],
                help="Upload your ratings/values data file"
            )
        
        if demographic_file and ratings_file:
            if st.button("📥 Load & Process Data", type="primary"):
                with st.spinner("Loading and preprocessing data..."):
                    try:
                        with open("temp_demo.csv", "wb") as f:
                            f.write(demographic_file.getbuffer())
                        with open("temp_ratings.csv", "wb") as f:
                            f.write(ratings_file.getbuffer())
                        
                        rec_system.load_data("temp_demo.csv", "temp_ratings.csv")
                        st.session_state.data_loaded = True
                        
                        os.remove("temp_demo.csv")
                        os.remove("temp_ratings.csv")
                        
                        st.success("✅ Data loaded and processed successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Users", rec_system.data_processor.num_users)
                        with col2:
                            st.metric("Items", rec_system.data_processor.num_items)
                        with col3:
                            st.metric("Ratings", len(rec_system.data_processor.ratings_data))
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("🎯 Step 2: Train Model")
            
            with st.expander("📋 Training Configuration"):
                col1, col2 = st.columns(2)
                with col1:
                    st.json({
                        "embedding_dims": DEFAULT_CONFIG['embedding_dims'],
                        "hidden_dims": DEFAULT_CONFIG['hidden_dims'],
                        "dropout": DEFAULT_CONFIG['dropout'],
                        "learning_rate": DEFAULT_CONFIG['learning_rate']
                    })
                with col2:
                    st.json({
                        "weight_decay": DEFAULT_CONFIG['weight_decay'],
                        "batch_size": DEFAULT_CONFIG['batch_size'],
                        "epochs": DEFAULT_CONFIG['epochs']
                    })
            
            if st.button("🚀 Train Model", type="primary"):
                training_results = train_model_with_real_progress(rec_system, DEFAULT_CONFIG['epochs'])
                
                if training_results:
                    st.session_state.model_trained = True
                    
                    rec_system.save_model('complete_dyslexia_model.pth')
                    
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                    
                    st.subheader("📈 Training Results")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Test RMSE", f"{training_results['test_metrics']['rmse']:.4f}")
                    with col2:
                        st.metric("Test R²", f"{training_results['test_metrics']['r2']:.4f}")
                    with col3:
                        st.metric("Precision@10", f"{training_results['test_metrics']['precision_10']:.4f}")
                    with col4:
                        st.metric("Recall@10", f"{training_results['test_metrics']['recall_10']:.4f}")
                    
                    if 'training_history' in training_results:
                        history_df = pd.DataFrame(training_results['training_history'])
                        st.line_chart(history_df[['loss', 'val_loss']])
    
    with tab2:
        st.header("👤 Get Personalized Recommendations")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first in the System Setup tab.")
            return
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the System Setup tab.")
            return
        
        st.subheader("🎯 Select User Type")
        
        user_type = st.selectbox(
            "Choose user type:",
            ["existing_user", "new_user_with_preference", "new_user_without_preference"],
            format_func=lambda x: {
                "new_user_with_preference": "🆕 New User with Preferences",
                "new_user_without_preference": "🆕 New User without Preferences",
                "existing_user": "👤 Existing User"
            }[x]
        )
        
        if user_type == "existing_user":
            st.subheader("👤 Existing User Recommendations")
            st.info("Enter your User ID to get personalized recommendations based on your preferences")
            
            user_id = st.number_input(
                "Your User ID",
                min_value=0,
                max_value=1204,
                value=0,
                step=1,
                help="Enter your user ID (0-1204)"
            )
            
            if user_id >= 0:
                user_info = get_user_info(rec_system, user_id) if st.session_state.model_trained else None
                
                if user_info and user_info is not None:
                    st.success(f"✅ User {user_id} found with {user_info['stats']['total_ratings']} ratings")
                elif user_id > 0:
                    st.error(f"❌ User {user_id} not found in the system")
            
            if st.button("🎯 Get My Recommendations", type="primary"):
                user_profile = {'id': user_id}
                
                with st.spinner("Analyzing your preferences and generating recommendations..."):
                    try:
                        recommendations_json = rec_system.get_recommendations(user_profile, top_k=10)
                        recommendations_data = json.loads(recommendations_json)
                        
                        if 'error' in recommendations_data:
                            st.error(f"Error: {recommendations_data['error']}")
                        else:
                            if user_info:
                                st.header(f"👤 User {user_id} Profile")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Ratings", user_info['stats']['total_ratings'])
                                with col2:
                                    st.metric("Average Rating", f"{user_info['stats']['avg_rating']}/5.0")
                                with col3:
                                    quality = user_info['preference_quality']
                                    quality_emoji = "🔥" if quality == 'high' else "👍" if quality == 'moderate' else "📈"
                                    st.metric("Preference Quality", f"{quality_emoji} {quality.title()}")
                                with col4:
                                    st.metric("Categories Explored", len(user_info['category_preferences']))
                                
                                if user_info['top_items']:
                                    st.subheader("⭐ Your Highly Rated Items")
                                    for item in user_info['top_items'][:5]:
                                        st.markdown(f"- **{item['item_code']}** ({item['category']}): {item['rating']}/5.0")
                                
                                if user_info['category_preferences']:
                                    st.subheader("📊 Your Category Preferences")
                                    for category, data in list(user_info['category_preferences'].items())[:10]:
                                        st.markdown(f"- **{category}**: {data['average_rating']}/5.0 ({data['items_rated']} items)")
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
        
        else:
            with st.form("user_recommendation_form"):
                st.subheader("📝 User Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Information**")
                    user_id = st.number_input(
                        "Assign User ID",
                        min_value=7000,
                        max_value=9999,
                        value=7000
                    )
                    
                    age = st.number_input("Age", min_value=5, max_value=80, value=25)
                    gender = st.selectbox("Gender", ["M", "F"])
                
                with col2:
                    st.markdown("**Learning Background**")
                    
                    diagnosis_timing = st.selectbox(
                        "When did you receive your dyslexia diagnosis?",
                        ['Elementari', 'Medie', 'Superiori (1° o 2° anno)', 'Superiori (3°, 4° o 5° anno)']
                    )
                    
                    has_other_difficulties = st.selectbox(
                        "Do you have other learning difficulties?",
                        ['No, solo dislessia', 'Si']
                    )
                    
                    other_difficulties_details = st.selectbox(
                        "If yes, which other difficulties?",
                        ['Altro','Difficoltà nel calcolo - Discalculia','Difficoltà nel calcolo - Discalculia, Altro',"Difficoltà nel calcolo - Discalculia, Difficoltà nell'ortografia - Disortografia","Difficoltà nel calcolo - Discalculia, Difficoltà nell'ortografia - Disortografia, Altro",'Difficoltà nel calcolo - Discalculia, Difficoltà nella scrittura - Disgrafia','Difficoltà nel calcolo - Discalculia, Difficoltà nella scrittura - Disgrafia, Altro',"Difficoltà nel calcolo - Discalculia, Difficoltà nella scrittura - Disgrafia, Difficoltà nell'ortografia - Disortografia","Difficoltà nel calcolo - Discalculia, Difficoltà nella scrittura - Disgrafia, Difficoltà nell'ortografia - Disortografia, Altro","Difficoltà nell'ortografia - Disortografia","Difficoltà nell'ortografia - Disortografia, Altro",'Difficoltà nella scrittura - Disgrafia','Difficoltà nella scrittura - Disgrafia, Altro',"Difficoltà nella scrittura - Disgrafia, Difficoltà nell'ortografia - Disortografia","Difficoltà nella scrittura - Disgrafia, Difficoltà nell'ortografia - Disortografia, Altro",'Nessuno']
                    ) if has_other_difficulties == "Si" else "Nessuno"
                    
                    family_history = st.selectbox(
                        "Family history of dyslexia?",
                        ["No", "Si"]
                    )
                
                preferences = []
                if user_type == "new_user_with_preference":
                    st.subheader("🎯 Learning Tool Preferences")
                    st.info("Rate your interest in different tool categories (1-5 scale)")
                    
                    unique_categories = sorted(set(CATEGORY_MAPPING.values()))
                    
                    pref_cols = st.columns(2)
                    for i, category in enumerate(unique_categories):
                        with pref_cols[i % 2]:
                            rating = st.slider(
                                f"{category}",
                                min_value=1,
                                max_value=5,
                                value=3,
                                key=f"pref_{category.replace(' ', '_')}"
                            )
                            if rating > 3:
                                preferences.append(category)
                
                submitted = st.form_submit_button("🎯 Get Recommendations", type="primary")
                
                if submitted:
                    user_profile = {
                        'id': user_id,
                        'age': age,
                        'gender': gender,
                        'diagnosis_timing': diagnosis_timing,
                        'has_other_difficulties': has_other_difficulties,
                        'other_difficulties_details': other_difficulties_details,
                        'family_history': family_history
                    }
                    
                    if user_type == "new_user_with_preference" and preferences:
                        user_profile['preferences'] = preferences
                    
                    with st.spinner("Generating personalized recommendations..."):
                        try:
                            recommendations_json = rec_system.get_recommendations(user_profile, top_k=10)
                            recommendations_data = json.loads(recommendations_json)
                            
                            if 'error' in recommendations_data:
                                st.error(f"Error: {recommendations_data['error']}")
                            else:
                                st.success("🌟 Recommendations Generated Successfully!")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("User ID", recommendations_data.get('user_id', 'N/A'))
                                with col2:
                                    st.metric("User Type", recommendations_data.get('user_type', 'N/A').title())
                                with col3:
                                    st.metric("Model Type", "Neural Collaborative Filtering")
                                
                                if 'recommendations' in recommendations_data:
                                    st.subheader("🎯 Your Personalized Recommendations")
                                    
                                    recommendations = recommendations_data['recommendations']
                                    
                                    for i, rec in enumerate(recommendations):
                                        with st.container():
                                            col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                                            
                                            with col1:
                                                st.markdown(f"**#{i+1}**")
                                            
                                            with col2:
                                                st.markdown(f"**{rec.get('item_code', 'Unknown')}**")
                                                st.caption(f"Category: {rec.get('category', 'Other')}")
                                            
                                            with col3:
                                                if user_type == "new_user_with_preference":
                                                    st.markdown("🎯 *Based on Preferences*")
                                                else:
                                                    st.markdown("📈 *Popular Item*")
                                            
                                            with col4:
                                                rating = rec.get('predicted_rating', 0)
                                                st.metric("Rating", f"{rating}/5.0")
                                            
                                            st.markdown("---")
                                    
                                    st.subheader("📊 Recommendation Analysis")
                                    
                                    rec_df = pd.DataFrame(recommendations)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if 'category' in rec_df.columns:
                                            category_counts = rec_df['category'].value_counts()
                                            st.bar_chart(category_counts)
                                            st.caption("Recommendations by category")
                                    
                                    with col2:
                                        if 'predicted_rating' in rec_df.columns:
                                            avg_rating = rec_df['predicted_rating'].astype(float).mean()
                                            st.metric("Average Predicted Rating", f"{avg_rating:.2f}/5.0")
                                            
                                            rating_dist = rec_df['predicted_rating'].astype(float).value_counts().sort_index()
                                            st.bar_chart(rating_dist)
                                            st.caption("Rating distribution")
                                    
                                    csv = rec_df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Recommendations",
                                        data=csv,
                                        file_name=f"dyslexia_recommendations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                        
                        except Exception as e:
                            st.error(f"Error generating recommendations: {str(e)}")
    
    with tab3:
        st.header("📊 System Analytics & Insights")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first to view analytics.")
            return
        
        st.subheader("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            with col1:
                st.metric("Total Users", rec_system.data_processor.num_users)
            with col2:
                st.metric("Total Items", rec_system.data_processor.num_items)
            with col3:
                st.metric("Total Ratings", len(rec_system.data_processor.ratings_data))
            with col4:
                avg_rating = rec_system.data_processor.ratings_data['rating'].mean() * 5
                st.metric("Average Rating", f"{avg_rating:.2f}/5.0")
            
            st.subheader("📊 Rating Distribution")
            ratings_scaled = rec_system.data_processor.ratings_data['rating'] * 5
            hist_data = np.histogram(ratings_scaled, bins=20)
            chart_data = pd.DataFrame({
                'Rating': [f"{hist_data[1][i]:.1f}" for i in range(len(hist_data[0]))],
                'Count': hist_data[0]
            })
            st.bar_chart(chart_data.set_index('Rating'))
            
            st.subheader("👥 User Activity")
            user_activity = rec_system.data_processor.ratings_data['user_id'].value_counts()
            st.write(f"Most active user: {user_activity.index[0]} with {user_activity.iloc[0]} ratings")
            st.write(f"Average ratings per user: {user_activity.mean():.1f}")
            
            st.subheader("🏷️ Category Analysis")
            
            category_stats = []
            for item_id, item_code in rec_system.data_processor.item_id_to_code.items():
                category = CATEGORY_MAPPING.get(item_code, 'Other')
                item_ratings = rec_system.data_processor.ratings_data[
                    rec_system.data_processor.ratings_data['item_id'] == item_id
                ]
                if len(item_ratings) > 0:
                    category_stats.append({
                        'Category': category,
                        'Item': item_code,
                        'Avg Rating': item_ratings['rating'].mean() * 5,
                        'Rating Count': len(item_ratings)
                    })
            
            if category_stats:
                category_df = pd.DataFrame(category_stats)
                
                category_avg = category_df.groupby('Category')['Avg Rating'].mean().sort_values(ascending=False)
                st.bar_chart(category_avg)
                st.caption("Average ratings by category")
                
                st.subheader("⭐ Top Rated Items")
                top_items = category_df.nlargest(10, 'Avg Rating')
                st.dataframe(top_items, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error displaying analytics: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🧠 Dyslexia Learning Tools Recommender | Powered by Neural Collaborative Filtering<br>
        Complete system for personalized dyslexia learning tool recommendations
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
