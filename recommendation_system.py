import numpy as np
import pandas as pd
import os
import json
import logging
import torch
import torch.optim as optim
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import DEFAULT_CONFIG, LOGGING_CONFIG, DEVICE, CATEGORY_MAPPING
from data_processor import DataProcessor
from model import NeuralCollaborativeFiltering, TrainingEarlyStopping
from hpo import HyperparameterOptimizer
from utils import train_model_epoch, evaluate_model, get_popular_items, validate_embedding_indices

class DyslexiaRecommendationSystem:
    """Comprehensive recommendation system for dyslexic learners providing personalized learning tool suggestions based on user profiles and collaborative filtering."""

    def __init__(self,
                 embedding_dims: int = None,
                 hidden_dims: List[int] = None,
                 dropout: float = None,
                 learning_rate: float = None,
                 weight_decay: float = None,
                 batch_size: int = None,
                 device: str = None):
        """Initialize recommendation system with configurable hyperparameters and component initialization."""
        config = DEFAULT_CONFIG.copy()
        self.embedding_dims = embedding_dims or config['embedding_dims']
        self.hidden_dims = hidden_dims or config['hidden_dims']
        self.dropout = dropout or config['dropout']
        self.learning_rate = learning_rate or config['learning_rate']
        self.weight_decay = weight_decay or config['weight_decay']
        self.batch_size = batch_size or config['batch_size']
        self.device = device or DEVICE

        self.data_processor = DataProcessor()
        self.model = None
        self.hpo_optimizer = None
        self.best_params = None

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging system for tracking system operations and debugging information."""
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
        """Load and preprocess demographic and rating data from CSV files into machine learning ready format."""
        try:
            self.logger.info("Loading data files...")
            if not os.path.exists(demographic_file) or not os.path.exists(ratings_file):
                raise FileNotFoundError("Required data files not found")

            demographic_df = pd.read_csv(demographic_file)
            ratings_df = pd.read_csv(ratings_file)

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
        """Execute automated hyperparameter optimization using Optuna framework to find optimal model configuration."""
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
        """Train neural collaborative filtering model with optional hyperparameter optimization results and comprehensive evaluation."""
        if self.data_processor.user_data is None:
            raise ValueError("Data must be loaded before training")

        epochs = epochs or DEFAULT_CONFIG['epochs']
        validation_split = validation_split or DEFAULT_CONFIG['validation_split']
        early_stopping_patience = early_stopping_patience or DEFAULT_CONFIG['early_stopping_patience']

        self.logger.info("Starting model training...")
        
        validate_embedding_indices(
            self.data_processor.ratings_data,
            self.data_processor.num_users,
            self.data_processor.num_items
        )

        if use_best_params and self.best_params is not None:
            self.logger.info("Using optimized hyperparameters from HPO")
            self.embedding_dims = self.best_params['embedding_dims']
            self.hidden_dims = self.best_params['hidden_dims']
            self.dropout = self.best_params['dropout']
            self.learning_rate = self.best_params['learning_rate']
            self.weight_decay = self.best_params['weight_decay']
            self.batch_size = self.best_params['batch_size']

        train_data, val_data, test_data = self.data_processor.split_data(
            self.data_processor.user_data,
            self.data_processor.item_data,
            self.data_processor.ratings_data,
            validation_split
        )

        num_users = self.data_processor.num_users
        num_items = self.data_processor.num_items
        user_feature_dim = self.data_processor.user_data.shape[1]
        item_feature_dim = self.data_processor.item_data.shape[1]

        self.logger.info(f"Model dimensions - Users: {num_users}, Items: {num_items}, "
                        f"User features: {user_feature_dim}, Item features: {item_feature_dim}")

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

        training_history = train_model_epoch(
            self.model, train_data, val_data, epochs, self.batch_size,
            optimizer, criterion, early_stopping, self.device
        )

        test_metrics = evaluate_model(self.model, test_data, self.batch_size, criterion, self.device)

        self.logger.info(f"Training completed. Test RMSE: {test_metrics['rmse']:.4f}")

        return {
            'training_history': training_history,
            'test_metrics': test_metrics
        }

    def get_recommendations(self, user_profile: Dict, top_k: int = 10) -> str:
        """Generate personalized recommendations for users based on profile analysis and return structured JSON response."""
        try:
            self._validate_user_profile(user_profile)
            
            if self.model is None:
                raise ValueError("Model must be trained before generating recommendations")

            user_id = user_profile.get('id')

            is_existing_user = False
            if user_id is not None:
                user_exists_in_ratings = len(self.data_processor.ratings_data[
                    self.data_processor.ratings_data['user_id'] == user_id
                ]) > 0
                is_existing_user = user_exists_in_ratings

            self.logger.info(f"User {user_id} - Existing: {is_existing_user}")

            if is_existing_user:
                recommendations = self._get_existing_user_recommendations(user_id, top_k)
            else:
                recommendations = self._get_new_user_recommendations(user_profile, top_k)

            result = {
                'user_id': user_id,
                'user_type': 'existing' if is_existing_user else 'new',
                'recommendations': recommendations,
                'model_info': {
                    'used_hpo': self.best_params is not None,
                    'best_params': self.best_params if self.best_params else 'default_params'
                },
                'debug_info': {
                    'user_exists_in_ratings': is_existing_user,
                    'total_users_in_system': self.data_processor.num_users,
                    'user_rating_count': len(self.data_processor.ratings_data[
                        self.data_processor.ratings_data['user_id'] == user_id
                    ]) if user_id is not None else 0
                },
                'timestamp': datetime.now().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return json.dumps({'error': str(e)})

    def _validate_user_profile(self, user_profile: Dict) -> None:
        """Validate user profile structure and preference categories against system constraints."""
        if not isinstance(user_profile, dict):
            raise ValueError("User profile must be a dictionary")

        if 'preferences' in user_profile:
            valid_categories = set(self.data_processor.category_names)
            user_prefs = set(user_profile['preferences'])
            invalid_prefs = user_prefs - valid_categories
            if invalid_prefs:
                raise ValueError(f"Invalid preferences: {invalid_prefs}")

    def _get_existing_user_recommendations(self, user_id: int, top_k: int) -> List[Dict]:
        """Generate personalized recommendations for existing users based on historical rating patterns and collaborative filtering."""
        try:
            user_ratings = self.data_processor.ratings_data[
                self.data_processor.ratings_data['user_id'] == user_id
            ]

            if len(user_ratings) == 0:
                return [{'error': f'User {user_id} not found in the system'}]

            user_avg_rating = user_ratings['rating'].mean()
            high_rated_threshold = max(user_avg_rating, 0.6)
            high_rated_items = user_ratings[user_ratings['rating'] >= high_rated_threshold]

            self.logger.info(f"User {user_id}: {len(high_rated_items)} high-rated items out of {len(user_ratings)} total")

            interacted_items = set(user_ratings['item_id'].values)
            all_items = set(range(self.data_processor.num_items))
            candidate_items = list(all_items - interacted_items)

            if len(candidate_items) == 0:
                return [{'info': 'User has interacted with all available items'}]

            category_recommendations = self._get_category_based_recommendations(
                high_rated_items, candidate_items, user_id
            )

            collaborative_recommendations = self._get_collaborative_recommendations_from_high_rated(
                high_rated_items, candidate_items, user_id
            )

            combined_recommendations = self._combine_personalized_recommendations(
                category_recommendations, collaborative_recommendations
            )

            if len(combined_recommendations) >= top_k:
                return combined_recommendations[:top_k]

            popular_items = get_popular_items(
                self.data_processor.ratings_data,
                top_k - len(combined_recommendations),
                self.data_processor.item_id_to_code
            )

            for item in popular_items:
                item['recommendation_type'] = 'popular_fallback'
                item['reason'] = 'Insufficient personalized recommendations'

            for item in combined_recommendations:
                item['recommendation_type'] = 'personalized'

            final_recommendations = combined_recommendations + popular_items

            if len(combined_recommendations) < top_k * 0.7:
                insight = {
                    'item_code': 'USER_INSIGHT',
                    'predicted_rating': 0,
                    'category': 'System Message',
                    'recommendation_type': 'insight',
                    'message': f'Based on your {len(high_rated_items)} high-rated items. Consider rating more items for better personalization.'
                }
                final_recommendations.insert(0, insight)

            return final_recommendations[:top_k + 1]

        except Exception as e:
            self.logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            return [{'error': f'Error generating recommendations: {str(e)}'}]

    def _get_category_based_recommendations(self, high_rated_items: pd.DataFrame, candidate_items: List[int], user_id: int) -> List[Dict]:
        """Analyze user's preferred categories from high-rated items and recommend similar category items with preference scoring."""
        category_preferences = {}
        category_item_counts = {}
        
        for _, rating_row in high_rated_items.iterrows():
            item_id = rating_row['item_id']
            rating = rating_row['rating'] * 5
            item_code = self.data_processor.item_id_to_code.get(item_id, f'ITEM_{item_id}')
            category = CATEGORY_MAPPING.get(item_code, 'Other')
            
            if category not in category_preferences:
                category_preferences[category] = []
                category_item_counts[category] = 0
            
            category_preferences[category].append(rating)
            category_item_counts[category] += 1

        category_scores = {}
        for category, ratings in category_preferences.items():
            avg_rating = sum(ratings) / len(ratings)
            item_count = category_item_counts[category]
            category_scores[category] = avg_rating * (1 + 0.1 * item_count)

        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        self.logger.info(f"User {user_id} preferred categories: {sorted_categories[:5]}")

        recommendations = []
        for category, score in sorted_categories:
            category_candidates = []
            for item_id in candidate_items:
                item_code = self.data_processor.item_id_to_code.get(item_id, f'ITEM_{item_id}')
                item_category = CATEGORY_MAPPING.get(item_code, 'Other')
                if item_category == category:
                    category_candidates.append(item_id)

            category_items_all = []
            for item_id, item_code in self.data_processor.item_id_to_code.items():
                if CATEGORY_MAPPING.get(item_code, 'Other') == category:
                    category_items_all.append(item_id)

            if category_items_all:
                category_ratings = self.data_processor.ratings_data[
                    self.data_processor.ratings_data['item_id'].isin(category_items_all)
                ]
                avg_category_rating = category_ratings['rating'].mean() * 5 if len(category_ratings) > 0 else 3.5
            else:
                avg_category_rating = 3.5

            for item_id in category_candidates[:5]:
                item_code = self.data_processor.item_id_to_code.get(item_id, f'ITEM_{item_id}')
                predicted_rating = min(score * 0.8, 5.0)
                
                recommendations.append({
                    'item_id': item_id,
                    'item_code': item_code,
                    'predicted_rating': round(predicted_rating, 2),
                    'category': category,
                    'source': 'category_preference',
                    'user_category_score': round(score, 2)
                })

        return sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)

    def _get_collaborative_recommendations_from_high_rated(self, high_rated_items: pd.DataFrame, candidate_items: List[int], user_id: int) -> List[Dict]:
        """Find users with similar high-rated item preferences and recommend items they enjoyed using collaborative filtering approach."""
        if len(high_rated_items) < 2:
            return []

        user_high_rated_item_ids = set(high_rated_items['item_id'].values)

        similar_users_scores = {}
        for item_id in user_high_rated_item_ids:
            item_ratings = self.data_processor.ratings_data[
                (self.data_processor.ratings_data['item_id'] == item_id) &
                (self.data_processor.ratings_data['rating'] >= 0.6) &
                (self.data_processor.ratings_data['user_id'] != user_id)
            ]

            for _, rating_row in item_ratings.iterrows():
                other_user_id = rating_row['user_id']
                rating = rating_row['rating']
                if other_user_id not in similar_users_scores:
                    similar_users_scores[other_user_id] = 0
                similar_users_scores[other_user_id] += rating

        top_similar_users = sorted(similar_users_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        if not top_similar_users:
            return []

        self.logger.info(f"User {user_id}: Found {len(top_similar_users)} similar users")

        collaborative_scores = {}
        for similar_user_id, similarity_score in top_similar_users:
            similar_user_ratings = self.data_processor.ratings_data[
                (self.data_processor.ratings_data['user_id'] == similar_user_id) &
                (self.data_processor.ratings_data['rating'] >= 0.6)
            ]

            for _, rating_row in similar_user_ratings.iterrows():
                item_id = rating_row['item_id']
                rating = rating_row['rating'] * 5

                if item_id in candidate_items:
                    if item_id not in collaborative_scores:
                        collaborative_scores[item_id] = 0
                    weighted_score = (similarity_score / len(top_similar_users)) * rating
                    collaborative_scores[item_id] += weighted_score

        recommendations = []
        for item_id, score in collaborative_scores.items():
            item_code = self.data_processor.item_id_to_code.get(item_id, f'ITEM_{item_id}')
            category = CATEGORY_MAPPING.get(item_code, 'Other')
            predicted_rating = min(score / 2, 5.0)

            recommendations.append({
                'item_id': item_id,
                'item_code': item_code,
                'predicted_rating': round(predicted_rating, 2),
                'category': category,
                'source': 'collaborative_filtering',
                'similarity_score': round(score, 2)
            })

        return sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)

    def _combine_personalized_recommendations(self, category_recs: List[Dict], collaborative_recs: List[Dict]) -> List[Dict]:
        """Merge category-based and collaborative filtering recommendations using weighted scoring for optimal recommendation diversity."""
        combined_scores = {}

        category_weight = 0.6
        collaborative_weight = 0.4

        for rec in category_recs:
            item_id = rec['item_id']
            if item_id not in combined_scores:
                combined_scores[item_id] = {**rec, 'combined_score': 0, 'sources': []}
            combined_scores[item_id]['combined_score'] += category_weight * rec['predicted_rating']
            combined_scores[item_id]['sources'].append('category')

        for rec in collaborative_recs:
            item_id = rec['item_id']
            if item_id not in combined_scores:
                combined_scores[item_id] = {**rec, 'combined_score': 0, 'sources': []}
            combined_scores[item_id]['combined_score'] += collaborative_weight * rec['predicted_rating']
            combined_scores[item_id]['sources'].append('collaborative')

        final_recommendations = []
        for item_id, rec_data in combined_scores.items():
            rec_data['predicted_rating'] = round(rec_data['combined_score'], 2)
            rec_data['recommendation_sources'] = ', '.join(rec_data['sources'])
            del rec_data['combined_score']
            del rec_data['sources']
            final_recommendations.append(rec_data)

        return sorted(final_recommendations, key=lambda x: x['predicted_rating'], reverse=True)

    def _get_new_user_recommendations(self, user_profile: Dict, top_k: int) -> List[Dict]:
        """Generate recommendations for new users using preference-based filtering or popularity-based fallback strategies."""
        if 'preferences' in user_profile and user_profile['preferences']:
            return self._get_preference_based_recommendations(user_profile, top_k)
        return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)

    def _get_preference_based_recommendations(self, user_profile: Dict, top_k: int) -> List[Dict]:
        """Create recommendations based on explicitly stated user preferences using category average ratings."""
        user_preferences = user_profile['preferences']

        if not hasattr(self.data_processor, 'item_data_with_categories'):
            self._create_item_category_mapping()

        preferred_items = []
        for category in user_preferences:
            if category in self.data_processor.category_names:
                category_items = []
                for item_id, item_code in self.data_processor.item_id_to_code.items():
                    if item_code in CATEGORY_MAPPING and CATEGORY_MAPPING[item_code] == category:
                        category_items.append(item_id)
                preferred_items.extend(category_items)

        preferred_items = list(set(preferred_items))

        if len(preferred_items) == 0:
            return get_popular_items(self.data_processor.ratings_data, top_k, self.data_processor.item_id_to_code)

        recommendations = []
        category_avg_ratings = self._get_category_average_ratings()

        for item_id in preferred_items:
            item_code = self.data_processor.item_id_to_code.get(item_id, f'ITEM_{item_id}')
            category = CATEGORY_MAPPING.get(item_code, 'Other')
            avg_rating = category_avg_ratings.get(category, 4.0)

            recommendations.append({
                'item_id': item_id,
                'item_code': item_code,
                'predicted_rating': round(avg_rating, 2),
                'category': category
            })

        sorted_recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)

        if len(sorted_recommendations) < top_k:
            popular_items = get_popular_items(
                self.data_processor.ratings_data,
                top_k - len(sorted_recommendations),
                self.data_processor.item_id_to_code
            )
            sorted_recommendations.extend(popular_items)

        return sorted_recommendations[:top_k]

    def get_user_information(self, user_id: int) -> Dict:
        """Retrieve comprehensive user analytics including rating statistics, top preferences, and category analysis."""
        try:
            user_ratings = self.data_processor.ratings_data[
                self.data_processor.ratings_data['user_id'] == user_id
            ]

            if len(user_ratings) == 0:
                return {'error': f'User {user_id} not found in the system'}

            user_stats = {
                'total_ratings': len(user_ratings),
                'average_rating': round(user_ratings['rating'].mean() * 5, 2),
                'min_rating': round(user_ratings['rating'].min() * 5, 2),
                'max_rating': round(user_ratings['rating'].max() * 5, 2)
            }

            high_rated = user_ratings[user_ratings['rating'] > 0.7]
            top_items = []
            for _, row in high_rated.nlargest(10, 'rating').iterrows():
                item_code = self.data_processor.item_id_to_code.get(row['item_id'], f"ITEM_{row['item_id']}")
                category = CATEGORY_MAPPING.get(item_code, 'Other')
                top_items.append({
                    'item_code': item_code,
                    'category': category,
                    'rating': round(row['rating'] * 5, 2)
                })

            category_preferences = {}
            for category in set(CATEGORY_MAPPING.values()):
                category_items = []
                for item_id, item_code in self.data_processor.item_id_to_code.items():
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
                'user_id': user_id,
                'statistics': user_stats,
                'highly_rated_items': top_items,
                'category_preferences': dict(sorted_categories[:10]),
                'total_categories_rated': len(category_preferences),
                'preference_quality': 'high' if user_stats['average_rating'] >= 3.5 else 'moderate' if user_stats['average_rating'] >= 2.5 else 'low'
            }

        except Exception as e:
            self.logger.error(f"Error getting user information for {user_id}: {str(e)}")
            return {'error': str(e)}

    def save_model(self, filepath: str) -> None:
        """Serialize trained model and complete system state to file for later restoration and deployment."""
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
                'family_categories': self.data_processor.family_categories,
                'num_users': self.data_processor.num_users,
                'num_items': self.data_processor.num_items
            },
            'preprocessed_data': {
                'user_data': self.data_processor.user_data,
                'item_data': self.data_processor.item_data,
                'ratings_data': self.data_processor.ratings_data
            },
            'hpo_results': {
                'best_params': self.best_params,
                'study_completed': self.hpo_optimizer is not None
            }
        }

        torch.save(model_state, filepath)
        self.logger.info(f"Model and data saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Restore trained model and system state from serialized file for immediate recommendation generation."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_state = torch.load(filepath, map_location=self.device)

        system_state = model_state['system_state']
        self.data_processor.item_id_to_code = system_state['item_id_to_code']
        self.data_processor.category_names = system_state['category_names']
        self.data_processor.diagnosis_categories = system_state['diagnosis_categories']
        self.data_processor.other_categories = system_state['other_categories']
        self.data_processor.answer_categories = system_state['answer_categories']
        self.data_processor.family_categories = system_state['family_categories']
        self.data_processor.num_users = system_state['num_users']
        self.data_processor.num_items = system_state['num_items']

        if 'preprocessed_data' in model_state:
            preprocessed_data = model_state['preprocessed_data']
            self.data_processor.user_data = preprocessed_data['user_data']
            self.data_processor.item_data = preprocessed_data['item_data']
            self.data_processor.ratings_data = preprocessed_data['ratings_data']
            self.logger.info("Preprocessed data restored successfully")
        else:
            self.logger.warning("No preprocessed data found in saved model")

        if 'hpo_results' in model_state:
            self.best_params = model_state['hpo_results']['best_params']

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
        """Create internal mapping structure between items and their categorical classifications for efficient lookup operations."""
        self.data_processor.item_data_with_categories = {}
        for item_id, item_code in self.data_processor.item_id_to_code.items():
            category = CATEGORY_MAPPING.get(item_code, 'Other')
            if category not in self.data_processor.item_data_with_categories:
                self.data_processor.item_data_with_categories[category] = []
            self.data_processor.item_data_with_categories[category].append(item_id)

    def _get_category_average_ratings(self) -> Dict[str, float]:
        """Calculate mean rating scores for each item category to support preference-based recommendation scoring."""
        category_ratings = {}
        for category in self.data_processor.category_names:
            category_items = []
            for item_id, item_code in self.data_processor.item_id_to_code.items():
                if CATEGORY_MAPPING.get(item_code, 'Other') == category:
                    category_items.append(item_id)

            if category_items:
                category_item_ratings = self.data_processor.ratings_data[
                    self.data_processor.ratings_data['item_id'].isin(category_items)
                ]['rating']
                
                if len(category_item_ratings) > 0:
                    avg_rating = category_item_ratings.mean() * 5
                    category_ratings[category] = avg_rating
                else:
                    category_ratings[category] = 4.0
            else:
                category_ratings[category] = 4.0

        return category_ratings
