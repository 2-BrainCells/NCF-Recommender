"""
Main Dyslexia Recommendation System
"""
import numpy as np
import pandas as pd
import os
import json
import logging
import torch
import torch.optim as optim
from datetime import datetime
from typing import Dict, List, Optional

from config import DEFAULT_CONFIG, LOGGING_CONFIG, DEVICE
from config import CATEGORY_MAPPING
from data_processor import DataProcessor
from model import NeuralCollaborativeFiltering, TrainingEarlyStopping
from hpo import HyperparameterOptimizer
from utils import train_model_epoch, evaluate_model, get_popular_items, get_item_category


class DyslexiaRecommendationSystem:
    """Main recommendation system class"""
    
    def __init__(self, 
                 embedding_dims: int = None,
                 hidden_dims: List[int] = None,
                 dropout: float = None,
                 learning_rate: float = None,
                 weight_decay: float = None,
                 batch_size: int = None,
                 device: str = None):
        
        # Use default config if not provided
        config = DEFAULT_CONFIG.copy()
        
        self.embedding_dims = embedding_dims or config['embedding_dims']
        self.hidden_dims = hidden_dims or config['hidden_dims']
        self.dropout = dropout or config['dropout']
        self.learning_rate = learning_rate or config['learning_rate']
        self.weight_decay = weight_decay or config['weight_decay']
        self.batch_size = batch_size or config['batch_size']
        self.device = device or DEVICE
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.model = None
        self.hpo_optimizer = None
        self.best_params = None
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.FileHandler(LOGGING_CONFIG['filename']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, demographic_file: str, ratings_file: str) -> None:
        """Load and preprocess all data"""
        try:
            self.logger.info("Loading data files...")
            
            if not os.path.exists(demographic_file) or not os.path.exists(ratings_file):
                raise FileNotFoundError("Required data files not found")
            
            demographic_df = pd.read_csv(demographic_file)
            ratings_df = pd.read_csv(ratings_file)
            
            # Process data using DataProcessor
            (self.data_processor.user_data, 
             self.data_processor.item_data, 
             self.data_processor.ratings_data) = self.data_processor.preprocess_data(
                demographic_df, ratings_df
            )
            
            self.logger.info(f"Data loaded successfully: {len(self.data_processor.user_data)} users, "
                           f"{len(self.data_processor.item_data)} items, {len(self.data_processor.ratings_data)} ratings")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def run_hyperparameter_optimization(self, 
                                       n_trials: int = 100,
                                       timeout: int = 3600,
                                       save_study: bool = True,
                                       study_name: str = 'dyslexia_ncf_optimization') -> Dict:
        """Run hyperparameter optimization"""
        
        if self.data_processor.user_data is None:
            raise ValueError("Data must be loaded before running HPO")
        
        self.hpo_optimizer = HyperparameterOptimizer(self.data_processor, self.device)
        study = self.hpo_optimizer.run_optimization(n_trials, timeout, save_study, study_name)
        
        self.best_params = self.hpo_optimizer.get_best_config()
        
        return {
            'best_value': study.best_value,
            'best_params': self.best_params,
            'n_trials': len(study.trials)
        }

    def train_model(self, 
                    epochs: int = None, 
                    validation_split: float = None, 
                    early_stopping_patience: int = None, 
                    use_best_params: bool = True) -> Dict:
        """Train the recommendation model"""

        if self.data_processor.user_data is None:
            raise ValueError("Data must be loaded before training")

        epochs = epochs or DEFAULT_CONFIG['epochs']
        validation_split = validation_split or DEFAULT_CONFIG['validation_split']
        early_stopping_patience = early_stopping_patience or DEFAULT_CONFIG['early_stopping_patience']

        self.logger.info("Starting model training...")

        # Use HPO results if available and requested
        if use_best_params and self.best_params is not None:
            self.logger.info("Using optimized hyperparameters from HPO")
            self.embedding_dims = self.best_params['embedding_dims']
            self.hidden_dims = self.best_params['hidden_dims']
            self.dropout = self.best_params['dropout']
            self.learning_rate = self.best_params['learning_rate']
            self.weight_decay = self.best_params['weight_decay']
            self.batch_size = self.best_params['batch_size']

        # Split data
        train_data, val_data, test_data = self.data_processor.split_data(
            self.data_processor.user_data,
            self.data_processor.item_data,
            self.data_processor.ratings_data,
            validation_split
        )

        # Use the actual counts from data processor
        num_users = self.data_processor.num_users
        num_items = self.data_processor.num_items
        user_feature_dim = self.data_processor.user_data.shape[1]
        item_feature_dim = self.data_processor.item_data.shape[1]

        self.logger.info(f"Model dimensions - Users: {num_users}, Items: {num_items}, "
                        f"User features: {user_feature_dim}, Item features: {item_feature_dim}")

        # Initialize model
        self.model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim,
            embedding_dims=self.embedding_dims,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            device=self.device
        )

        self.model.to(self.device)

        # Setup training
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        early_stopping = TrainingEarlyStopping(
            patience=early_stopping_patience, 
            delta=0.0002
        )

        # Train model
        training_history = train_model_epoch(
            self.model, train_data, val_data, epochs, self.batch_size,
            optimizer, criterion, early_stopping, self.device
        )

        # Evaluate on test set
        test_metrics = evaluate_model(self.model, test_data, self.batch_size, criterion, self.device)

        self.logger.info(f"Training completed. Test RMSE: {test_metrics['rmse']:.4f}")

        return {
            'training_history': training_history,
            'test_metrics': test_metrics
        }


    def get_recommendations(self, user_profile: Dict, top_k: int = 10) -> str:
        """Get recommendations for a user and return as JSON"""
        
        try:
            self._validate_user_profile(user_profile)
            
            if self.model is None:
                raise ValueError("Model must be trained before generating recommendations")
            
            self.logger.info(f"Generating recommendations for user: {user_profile.get('id', 'new_user')}")
            
            # Check if user exists in system
            user_id = user_profile.get('id')
            is_existing_user = user_id is not None and user_id < len(self.data_processor.user_data.drop_duplicates())
            
            if is_existing_user:
                recommendations = self._get_existing_user_recommendations(user_id, top_k)
            else:
                recommendations = self._get_new_user_recommendations(user_profile, top_k)
            
            # Format as JSON
            result = {
                'user_id': user_id,
                'user_type': 'existing' if is_existing_user else 'new',
                'recommendations': recommendations,
                'model_info': {
                    'used_hpo': self.best_params is not None,
                    'best_params': self.best_params if self.best_params else 'default_params'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return json.dumps({'error': str(e)})

    def _validate_user_profile(self, user_profile: Dict) -> None:
        """Validate user profile data"""
        
        if not isinstance(user_profile, dict):
            raise ValueError("User profile must be a dictionary")
        
        # For new users, validate required fields
        if 'preferences' in user_profile:
            valid_categories = set(self.data_processor.category_names)
            user_prefs = set(user_profile['preferences'])
            invalid_prefs = user_prefs - valid_categories
            if invalid_prefs:
                raise ValueError(f"Invalid preferences: {invalid_prefs}")

    def _get_existing_user_recommendations(self, user_id: int, top_k: int) -> List[Dict]:
        """Get recommendations for existing user"""
        
        # Get user's interaction history
        user_ratings = self.data_processor.ratings_data[self.data_processor.ratings_data['user_id'] == user_id]
        interacted_items = set(user_ratings['item_id'].values)
        
        # Get all items user hasn't interacted with
        all_items = set(range(len(self.data_processor.item_data.drop_duplicates())))
        candidate_items = list(all_items - interacted_items)
        
        if len(candidate_items) == 0:
            return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)
        
        # Simple prediction for candidate items (would need more implementation)
        # For now, return popular items
        return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)

    def _get_new_user_recommendations(self, user_profile: Dict, top_k: int) -> List[Dict]:
        """Get recommendations for new user"""
        
        # If user has preferences, use them
        if 'preferences' in user_profile and user_profile['preferences']:
            return self._get_preference_based_recommendations(user_profile, top_k)
        
        # Otherwise, recommend popular items
        return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)
    
    def _get_preference_based_recommendations(self, user_profile: Dict, top_k: int) -> List[Dict]:
        """Get recommendations based on user preferences"""
        
        user_preferences = user_profile['preferences']
        
        # Find items in preferred categories
        preferred_items = []
        
        # Create a temporary item data with categories if not exists
        if not hasattr(self.data_processor, 'item_data_with_categories'):
            self._create_item_category_mapping()
        
        for category in user_preferences:
            if category in self.data_processor.category_names:
                # Find items that belong to this category
                category_items = []
                for item_id, item_code in self.data_processor.item_id_to_code.items():
                    if item_code in CATEGORY_MAPPING and CATEGORY_MAPPING[item_code] == category:
                        category_items.append(item_id)
                preferred_items.extend(category_items)
        
        # Remove duplicates
        preferred_items = list(set(preferred_items))
        
        if len(preferred_items) == 0:
            return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)
        
        # For preference-based recommendations, use the category average ratings
        recommendations = []
        category_avg_ratings = self._get_category_average_ratings()
        
        for item_id in preferred_items:
            item_code = self.data_processor.item_id_to_code.get(item_id, f'ITEM_{item_id}')
            category = CATEGORY_MAPPING.get(item_code, 'Other')
            
            # Get average rating for this category, default to 4.0
            avg_rating = category_avg_ratings.get(category, 4.0)
            
            recommendations.append({
                'item_id': item_id,
                'item_code': item_code,
                'predicted_rating': round(avg_rating, 2),
                'category': category
            })
        
        # Sort by predicted rating
        sorted_recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        
        # If we don't have enough recommendations, fill with popular items
        if len(sorted_recommendations) < top_k:
            popular_items = get_popular_items(
                self.data_processor.ratings_data, 
                top_k - len(sorted_recommendations), 
                self.data_processor.item_id_to_code
            )
            sorted_recommendations.extend(popular_items)
        
        return sorted_recommendations[:top_k]

    def save_model(self, filepath: str) -> None:
        """Save the trained model and system state"""
        
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'embedding_dims': self.embedding_dims,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
                'num_users': self.model.num_users,
                'num_items': self.model.num_items,
                'user_feature_dim': self.model.user_feature_dim,
                'item_feature_dim': self.model.item_feature_dim
            },
            'system_state': {
                'item_id_to_code': self.data_processor.item_id_to_code,
                'category_names': self.data_processor.category_names,
                'diagnosis_categories': self.data_processor.diagnosis_categories,
                'other_categories': self.data_processor.other_categories,
                'answer_categories': self.data_processor.answer_categories,
                'family_categories': self.data_processor.family_categories
            },
            'hpo_results': {
                'best_params': self.best_params,
                'study_completed': self.hpo_optimizer is not None
            }
        }
        
        torch.save(model_state, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a saved model and system state"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_state = torch.load(filepath, map_location=self.device)
        
        # Restore system state
        system_state = model_state['system_state']
        self.data_processor.item_id_to_code = system_state['item_id_to_code']
        self.data_processor.category_names = system_state['category_names']
        self.data_processor.diagnosis_categories = system_state['diagnosis_categories']
        self.data_processor.other_categories = system_state['other_categories']
        self.data_processor.answer_categories = system_state['answer_categories']
        self.data_processor.family_categories = system_state['family_categories']
        
        # Restore HPO results if available
        if 'hpo_results' in model_state:
            self.best_params = model_state['hpo_results']['best_params']
        
        # Recreate and load model
        config = model_state['model_config']
        self.model = NeuralCollaborativeFiltering(
            num_users=config['num_users'],
            num_items=config['num_items'],
            user_feature_dim=config['user_feature_dim'],
            item_feature_dim=config['item_feature_dim'],
            embedding_dims=config['embedding_dims'],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
            device=self.device
        )
        
        self.model.load_state_dict(model_state['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Model loaded from {filepath}")
        
    def _create_item_category_mapping(self):
        """Create mapping between items and categories"""
        self.data_processor.item_data_with_categories = {}
        for item_id, item_code in self.data_processor.item_id_to_code.items():
            category = CATEGORY_MAPPING.get(item_code, 'Other')
            if category not in self.data_processor.item_data_with_categories:
                self.data_processor.item_data_with_categories[category] = []
            self.data_processor.item_data_with_categories[category].append(item_id)
            
    def _get_category_average_ratings(self) -> Dict[str, float]:
        """Get average ratings per category"""
        category_ratings = {}
        
        for category in self.data_processor.category_names:
            # Find items in this category
            category_items = []
            for item_id, item_code in self.data_processor.item_id_to_code.items():
                if CATEGORY_MAPPING.get(item_code, 'Other') == category:
                    category_items.append(item_id)
            
            if category_items:
                # Get ratings for items in this category
                category_item_ratings = self.data_processor.ratings_data[
                    self.data_processor.ratings_data['item_id'].isin(category_items)
                ]['rating']
                
                if len(category_item_ratings) > 0:
                    # Convert back to 1-5 scale and get average
                    avg_rating = category_item_ratings.mean() * 5
                    category_ratings[category] = avg_rating
                else:
                    category_ratings[category] = 4.0  # Default rating
            else:
                category_ratings[category] = 4.0  # Default rating
        
        return category_ratings

    def _get_existing_user_recommendations(self, user_id: int, top_k: int) -> List[Dict]:
        """Get recommendations for existing user"""
        
        # Get user's interaction history
        user_ratings = self.data_processor.ratings_data[self.data_processor.ratings_data['user_id'] == user_id]
        interacted_items = set(user_ratings['item_id'].values)
        
        # Get all items user hasn't interacted with
        all_items = set(range(self.data_processor.num_items))
        candidate_items = list(all_items - interacted_items)
        
        if len(candidate_items) == 0:
            return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)
        
        # For existing users, we can use the model to predict ratings
        if len(candidate_items) > 0:
            recommendations = self._predict_ratings_for_user(user_id, candidate_items, top_k)
            return recommendations
        
        # Fallback to popular items
        return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)

    def _predict_ratings_for_user(self, user_id: int, candidate_items: List[int], top_k: int) -> List[Dict]:
        """Predict ratings for candidate items for an existing user"""
        
        # Get user features
        user_data_sample = self.data_processor.user_data.iloc[0:1]  # Get structure
        user_features = torch.FloatTensor(user_data_sample.values).to(self.device)
        
        predictions = []
        
        for item_id in candidate_items[:min(100, len(candidate_items))]:  # Limit to prevent memory issues
            # Get item features
            item_data_sample = self.data_processor.item_data.iloc[0:1]  # Get structure
            item_features = torch.FloatTensor(item_data_sample.values).to(self.device)
            
            # Create tensors for prediction
            user_id_tensor = torch.IntTensor([user_id]).to(self.device)
            item_id_tensor = torch.IntTensor([item_id]).to(self.device)
            
            # Predict rating
            self.model.eval()
            with torch.no_grad():
                try:
                    predicted_rating = self.model(user_id_tensor, item_id_tensor, user_features, item_features)
                    predicted_rating = predicted_rating.cpu().numpy()[0] * 5  # Convert to 1-5 scale
                    
                    item_code = self.data_processor.item_id_to_code.get(item_id, f'ITEM_{item_id}')
                    category = CATEGORY_MAPPING.get(item_code, 'Other')
                    
                    predictions.append({
                        'item_id': item_id,
                        'item_code': item_code,
                        'predicted_rating': round(predicted_rating, 2),
                        'category': category
                    })
                except Exception as e:
                    # Skip items that cause errors
                    continue
        
        # Sort by predicted rating
        sorted_predictions = sorted(predictions, key=lambda x: x['predicted_rating'], reverse=True)
        
        # If we don't have enough predictions, fill with popular items
        if len(sorted_predictions) < top_k:
            popular_items = get_popular_items(
                self.data_processor.ratings_data, 
                top_k - len(sorted_predictions), 
                self.data_processor.item_id_to_code
            )
            sorted_predictions.extend(popular_items)
        
        return sorted_predictions[:top_k]
    
    def get_item_category(item_id: int, item_data: pd.DataFrame, category_names: List[str], item_id_to_code: Dict) -> str:
        """Get category for an item"""
        item_code = item_id_to_code.get(item_id, f'ITEM_{item_id}')
        return CATEGORY_MAPPING.get(item_code, 'Other')
