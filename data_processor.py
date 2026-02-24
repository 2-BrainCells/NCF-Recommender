import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging
from config import CATEGORY_MAPPING, DATA_CONFIG

class DataProcessor:
    """Comprehensive data preprocessing pipeline for dyslexia recommendation system handling user demographics, item catalogs, and rating matrices."""
    
    def __init__(self):
        """Initialize data processor with empty containers for processed data and category mappings."""
        self.logger = logging.getLogger(__name__)
        self.item_id_to_code = {}
        self.category_names = []
        self.diagnosis_categories = []
        self.other_categories = []
        self.answer_categories = []
        self.family_categories = []
        self.num_users = 0
        self.num_items = 0

    def create_item_data(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Extract item information from ratings dataframe and create structured item catalog with category mappings."""
        item_columns = ratings_df.columns[12:]
        item_data = pd.DataFrame({
            'item_id': range(len(item_columns)),
            'item_code': item_columns,
            'category': [CATEGORY_MAPPING.get(col, 'Other') for col in item_columns]
        })
        
        self.item_id_to_code = dict(zip(item_data['item_id'], item_data['item_code']))
        self.category_names = sorted(item_data['category'].unique())
        self.num_items = len(item_columns)
        
        return item_data

    def create_ratings_data(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw ratings matrix by extracting ONLY observed explicit feedback and dropping non-interactions."""
        # Align with demographics by skipping the first row, use cols 12+ for tools
        ratings_matrix = ratings_df.iloc[1:, 12:].copy()
        
        # CRITICAL FIX 1: Replace 0, '0', 'NC', 'NSU', ' ' with np.nan to avoid penalizing unseen items
        ratings_matrix = ratings_matrix.replace(['NC', 'NSU', ' ', 0, '0', 0.0], np.nan)
        ratings_matrix = ratings_matrix.apply(pd.to_numeric, errors='coerce')

        self.num_users = ratings_matrix.shape[0]
        
        # Reset index before assigning user_id to map exactly to user_data's indices
        ratings_matrix.reset_index(drop=True, inplace=True)
        ratings_matrix.insert(0, 'user_id', ratings_matrix.index)

        ratings_data = ratings_matrix.melt(
            id_vars=['user_id'],
            var_name='item_code',
            value_name='rating'
        )

        # Drop missing values to create a true sparse matrix (Fixes 94% density issue)
        ratings_data = ratings_data.dropna(subset=['rating'])
        ratings_data['rating'] = ratings_data['rating'] / DATA_CONFIG['rating_scale']
        ratings_data.reset_index(drop=True, inplace=True)

        unique_item_codes = list(ratings_matrix.columns[1:])
        code_to_id = {code: idx for idx, code in enumerate(unique_item_codes)}
        ratings_data['item_id'] = ratings_data['item_code'].map(code_to_id)

        ratings_data = ratings_data[['user_id', 'item_id', 'rating']].round(2)
        ratings_data['user_id'] = ratings_data['user_id'].astype(int)
        ratings_data['item_id'] = ratings_data['item_id'].astype(int)

        return ratings_data

    def create_user_data(self, demographic_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Transform demographics and P1-P12 learning difficulties into structured user profiles."""
        # 1. Process standard demographics
        birth_year_mode = demographic_df['Anno di nascita'].mode()[0]
        demographic_df['Anno di nascita'] = demographic_df['Anno di nascita'].fillna(birth_year_mode)

        user_data = demographic_df.iloc[1:, :20].copy()
        user_data = user_data.rename(columns={
            'Anno di nascita': 'birth_year',
            'Sesso': 'gender',
            'Quale Corso di Laurea frequenti o hai frequentato?': 'degree',
            'Quando hai ricevuto la diagnosi di DSA?': 'diagnosis_timing',
            'Oltre alla dislessia hai altre difficoltà specifiche di apprendimento?': 'has_other_difficulties',
            '*SE HAI RISPOSTO SÌ ALLA PRECEDENTE DOMANDA* - Quali altre difficoltà hai oltre alla dislessia?': 'other_difficulties_details',
            'Hai qualche familiare dislessico?': 'family_history'
        })

        user_data = user_data[[
            'birth_year', 'gender', 'diagnosis_timing',
            'has_other_difficulties', 'other_difficulties_details', 'family_history'
        ]]

        current_year = datetime.now().year
        user_data['age'] = current_year - pd.to_numeric(user_data['birth_year'], errors='coerce').fillna(current_year-20)
        user_data = user_data.drop('birth_year', axis=1)
        user_data['other_difficulties_details'] = user_data['other_difficulties_details'].fillna('Nessuno')
        
        user_data.reset_index(drop=True, inplace=True)

        # 2. CRITICAL FIX 2: Extract P1-P12 Learning Difficulties
        # Align rows by skipping the first row of ratings_df
        p_features = ratings_df.iloc[1:, :12].copy()
        
        # Convert to numeric, replace non-numeric with NaN, fill with 0 (assuming 0 = no reported difficulty)
        p_features = p_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        p_features.reset_index(drop=True, inplace=True)
        
        # Prefix columns to easily identify them as learning difficulty metrics
        p_features.columns = [f"learning_diff_{col}" for col in p_features.columns]

        # 3. Concatenate Demographics + P1-P12
        user_data = pd.concat([user_data, p_features], axis=1)
        user_data.insert(0, 'user_id', user_data.index)

        self.diagnosis_categories = sorted(user_data['diagnosis_timing'].astype(str).unique())
        self.other_categories = sorted(user_data['has_other_difficulties'].astype(str).unique())
        self.answer_categories = sorted(user_data['other_difficulties_details'].astype(str).unique())
        self.family_categories = sorted(user_data['family_history'].astype(str).unique())

        return user_data

    def encode_user_features(self, user_data: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical user features into numerical format using one-hot encoding for machine learning compatibility."""
        categorical_columns = ['gender', 'diagnosis_timing', 'has_other_difficulties',
                              'other_difficulties_details', 'family_history']
        return pd.get_dummies(user_data, columns=categorical_columns).astype(float)

    def encode_item_features(self, item_data: pd.DataFrame) -> pd.DataFrame:
        """Transform item categories into binary feature vectors for neural network input processing."""
        categories_df = pd.get_dummies(item_data['category'])
        item_encoded = pd.concat([item_data[['item_id']], categories_df], axis=1)
        return item_encoded.astype(float)

    def extract_user_category_preferences(self, user_data: pd.DataFrame,
                                        item_data: pd.DataFrame,
                                        ratings_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate user preference scores for each item category based on historical rating patterns."""
        features_df = user_data.copy()

        def exponential_penalty(count, decay_factor=DATA_CONFIG['exponential_decay_factor']):
            """Apply exponential decay penalty for users with limited interaction history."""
            return 1 / np.exp(decay_factor * count)

        for category in self.category_names:
            category_items = item_data[item_data[category] == 1]['item_id']
            category_ratings = ratings_data[ratings_data['item_id'].isin(category_items)]
            
            user_category_stats = category_ratings.groupby('user_id')['rating'].agg(['mean', 'count']).reset_index()
            user_category_stats.columns = ['user_id', f'avg_rating_{category}', f'rating_count_{category}']
            
            features_df = pd.merge(features_df, user_category_stats, on='user_id', how='left')

            features_df[f'avg_rating_{category}'] = (
                (1 - exponential_penalty(features_df[f'rating_count_{category}'].fillna(0))) *
                features_df[f'avg_rating_{category}'].fillna(0)
            )

        return features_df

    def extend_user_item_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame,
                            ratings_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align user and item feature matrices to match the structure and ordering of the ratings data."""
        unique_users = ratings_data['user_id'].unique()
        unique_items = ratings_data['item_id'].unique()

        user_data_filtered = user_data[user_data['user_id'].isin(unique_users)]
        item_data_filtered = item_data[item_data['item_id'].isin(unique_items)]

        user_data_extended = pd.merge(ratings_data[['user_id']], user_data_filtered, on='user_id', how='left')
        item_data_extended = pd.merge(ratings_data[['item_id']], item_data_filtered, on='item_id', how='left')

        # CRITICAL FIX 3: PyTorch safety net. Fill any NaNs created by the merge with 0 to prevent network poisoning
        user_data_extended = user_data_extended.fillna(0)
        item_data_extended = item_data_extended.fillna(0)

        user_data_extended = user_data_extended.drop('user_id', axis=1)
        item_data_extended = item_data_extended.drop('item_id', axis=1)

        return user_data_extended, item_data_extended

    def split_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame,
                  ratings_data: pd.DataFrame, validation_split: float = 0.2) -> Tuple:
        """Partition dataset into training, validation, and test sets."""
        train_ratings, temp_ratings = train_test_split(ratings_data, test_size=0.2, random_state=42)
        val_ratings, test_ratings = train_test_split(temp_ratings, test_size=0.5, random_state=42)

        train_indices = train_ratings.index
        val_indices = val_ratings.index
        test_indices = test_ratings.index

        X_user_train = user_data.iloc[train_indices].values
        X_user_val = user_data.iloc[val_indices].values
        X_user_test = user_data.iloc[test_indices].values

        X_item_train = item_data.iloc[train_indices].values
        X_item_val = item_data.iloc[val_indices].values
        X_item_test = item_data.iloc[test_indices].values

        y_train = train_ratings.values
        y_val = val_ratings.values
        y_test = test_ratings.values

        return (
            (X_user_train, X_item_train, y_train),
            (X_user_val, X_item_val, y_val),
            (X_user_test, X_item_test, y_test)
        )

    def preprocess_data(self, demographic_df: pd.DataFrame,
                       ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute complete data preprocessing pipeline from raw CSV files to machine learning ready tensors."""
        self.logger.info("Starting data preprocessing...")

        item_data = self.create_item_data(ratings_df)
        self.logger.info(f"Created item data: {len(item_data)} items")

        ratings_data = self.create_ratings_data(ratings_df)
        self.logger.info(f"Created ratings data: {len(ratings_data)} ratings")

        # Now passes both dataframes so P1-P12 can be scraped
        user_data = self.create_user_data(demographic_df, ratings_df)
        self.logger.info(f"Created user data: {len(user_data)} users")

        user_data = self.encode_user_features(user_data)
        item_data = self.encode_item_features(item_data)
        user_data = self.extract_user_category_preferences(user_data, item_data, ratings_data)

        user_data, item_data = self.extend_user_item_data(user_data, item_data, ratings_data)

        self.logger.info(f"Final data shapes - Users: {user_data.shape}, Items: {item_data.shape}, Ratings: {ratings_data.shape}")

        return user_data, item_data, ratings_data