"""
Utility functions for Dyslexia Recommendation System
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
from typing import Dict, List, Tuple
import sys
import os


class HiddenPrints:
    """Context manager to suppress print statements during HPO"""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def precision_recall_at_k(y_preds: np.ndarray, y_true: np.ndarray, X_test_users: np.ndarray, k: int = 5, threshold: float = 0.6) -> Tuple[float, float]:
    """Calculate Precision@K and Recall@K for recommendation system"""
    
    unique_users = np.unique(X_test_users, axis=0)
    precisions = []
    recalls = []
    
    for user in unique_users:
        user_indices = np.where((X_test_users == user).all(axis=1))[0]
        if len(user_indices) == 0:
            continue
        
        user_preds = y_preds[user_indices]
        user_true = y_true[user_indices]
        
        actual_k = min(k, len(user_preds))
        if actual_k == 0:  # Add this check
            continue
            
        top_k_indices = np.argsort(user_preds)[-actual_k:][::-1]
        top_k_true = user_true[top_k_indices]
        
        relevant_in_top_k = np.sum(top_k_true >= threshold)
        total_relevant = np.sum(user_true >= threshold)
        
        precision = relevant_in_top_k / actual_k if actual_k > 0 else 0
        recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0

def init_data(input_data: Tuple, y: np.ndarray = None, device: str = 'cpu') -> Tuple:
    """Initialize data tensors for model input"""
    
    X_user_id, X_item_id, X_user, X_item = input_data
    
    X_user = torch.FloatTensor(X_user).to(device, non_blocking=True)
    X_item = torch.FloatTensor(X_item).to(device, non_blocking=True)
    X_user_id = torch.IntTensor(X_user_id).to(device, non_blocking=True)
    X_item_id = torch.IntTensor(X_item_id).to(device, non_blocking=True)
    y = torch.FloatTensor(y).to(device, non_blocking=True) if y is not None else None
    
    return X_user_id, X_item_id, X_user, X_item, y


def evaluate_model(model: nn.Module, 
                  test_data: Tuple, 
                  batch_size: int, 
                  criterion: nn.Module,
                  device: str = 'cpu') -> Dict:
    """Evaluate model performance"""
    
    X_user_test, X_item_test, y_test = test_data
    
    test_dataset = TensorDataset(
        torch.IntTensor(y_test[:, 0]),
        torch.IntTensor(y_test[:, 1]),
        torch.FloatTensor(X_user_test),
        torch.FloatTensor(X_item_test),
        torch.FloatTensor(y_test[:, 2])
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for user_ids, item_ids, user_features, item_features, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            user_features = user_features.to(device)
            item_features = item_features.to(device)
            ratings = ratings.to(device)
            
            predictions = model(user_ids, item_ids, user_features, item_features)
            loss = criterion(predictions, ratings)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    val_loss = total_loss / len(test_loader)
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = sqrt(mse)
    
    # Calculate precision@10 and recall@10
    precision_10, recall_10 = precision_recall_at_k(
        predictions, targets, X_user_test, k=10, threshold=0.6
    )
    
    return {
        'val_loss': val_loss,
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'precision_10': precision_10,
        'recall_10': recall_10
    }


def train_model_epoch(model: nn.Module,
                     train_data: Tuple,
                     val_data: Tuple,
                     epochs: int,
                     batch_size: int,
                     optimizer: torch.optim.Optimizer,
                     criterion: nn.Module,
                     early_stopping = None,
                     device: str = 'cpu') -> Dict:
    """Train model for given epochs"""
    
    X_user_train, X_item_train, y_train = train_data
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.IntTensor(y_train[:, 0]),  # user_ids
        torch.IntTensor(y_train[:, 1]),  # item_ids
        torch.FloatTensor(X_user_train),
        torch.FloatTensor(X_item_train),
        torch.FloatTensor(y_train[:, 2])  # ratings
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'r2': [],
        'mae': [],
        'mse': [],
        'rmse': [],
        'precision_10': [],
        'recall_10': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (user_ids, item_ids, user_features, item_features, ratings) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            user_features = user_features.to(device)
            item_features = item_features.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids, user_features, item_features)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        history['loss'].append(avg_train_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            val_metrics = evaluate_model(model, val_data, batch_size, criterion, device)
            
            for key, value in val_metrics.items():
                history[key].append(value)
            
            if early_stopping:
                early_stopping(val_metrics['val_loss'], model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
    
    return history


def get_popular_items(ratings_data: pd.DataFrame, top_k: int, item_id_to_code: Dict) -> List[Dict]:
    """Get popular items as fallback recommendations"""
    
    # Calculate item popularity based on average ratings
    item_popularity = ratings_data.groupby('item_id')['rating'].agg(['mean', 'count']).reset_index()
    item_popularity['popularity_score'] = item_popularity['mean'] * np.log1p(item_popularity['count'])
    item_popularity = item_popularity.sort_values('popularity_score', ascending=False)
    
    popular_items = []
    for _, item in item_popularity.head(top_k).iterrows():
        popular_items.append({
            'item_id': int(item['item_id']),
            'item_code': item_id_to_code.get(item['item_id'], f'ITEM_{item["item_id"]}'),
            'predicted_rating': round(item['mean'] * 5, 2),  # Convert to 1-5 scale
            'category': 'Popular',
            'popularity_score': round(item['popularity_score'], 2)
        })
    
    return popular_items


def get_item_category(item_id: int, item_data: pd.DataFrame, category_names: List[str]) -> str:
    """Get category for an item"""
    for category in category_names:
        if item_data.iloc[item_id][category] == 1:
            return category
    return 'Other'

def validate_data_consistency(ratings_data: pd.DataFrame, num_users: int, num_items: int) -> bool:
    """Validate that the data is consistent with model dimensions"""
    
    max_user_id = ratings_data['user_id'].max()
    max_item_id = ratings_data['item_id'].max()
    
    if max_user_id >= num_users:
        raise ValueError(f"Max user_id ({max_user_id}) exceeds num_users ({num_users})")
    
    if max_item_id >= num_items:
        raise ValueError(f"Max item_id ({max_item_id}) exceeds num_items ({num_items})")
    
    return True
