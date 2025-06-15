"""
Neural Collaborative Filtering Model for Dyslexia Recommendation System
"""
import torch
import torch.nn as nn
from typing import List, Optional
import optuna

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model combining Matrix Factorization and MLP
    """
    
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 user_feature_dim: int, 
                 item_feature_dim: int, 
                 embedding_dims: int, 
                 hidden_dims: List[int],
                 dropout: float = 0.1, 
                 device: str = 'cpu', 
                 trial: Optional[optuna.Trial] = None):
        
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.device = device
        
        # Embedding layers
        self.user_embedding_mlp = nn.Embedding(num_users + 1, embedding_dims)
        self.item_embedding_mlp = nn.Embedding(num_items + 1, embedding_dims)
        self.user_embedding_mf = nn.Embedding(num_users + 1, embedding_dims)
        self.item_embedding_mf = nn.Embedding(num_items + 1, embedding_dims)
        
        # Feature transformation layers
        self.user_feature_transform = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dims * 2),
            nn.ReLU(),
            nn.Linear(embedding_dims * 2, embedding_dims)
        )
        
        self.item_feature_transform = nn.Sequential(
            nn.Linear(item_feature_dim, embedding_dims * 2),
            nn.ReLU(),
            nn.Linear(embedding_dims * 2, embedding_dims)
        )
        
        # MLP layers
        mlp_layers = []
        input_dim = 4 * embedding_dims
        
        if trial is None:
            # Use provided hidden_dims
            for hidden_dim in hidden_dims:
                mlp_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                input_dim = hidden_dim
        else:
            # HPO mode: suggest architecture
            n_layers = trial.suggest_int('n_layers', 2, 5)
            for i in range(n_layers):
                hidden_dim = trial.suggest_categorical(f'hidden_dim_layer_{i}', [128, 256, 512, 1024])
                mlp_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                input_dim = hidden_dim
        
        mlp_layers.extend([
            nn.Linear(input_dim, embedding_dims),
            nn.ReLU()
        ])
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.prediction_layer = nn.Linear(2 * embedding_dims, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
        
        for module in self.mlp_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.01)
        
        nn.init.xavier_uniform_(self.prediction_layer.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, 
                user_ids: torch.Tensor, 
                item_ids: torch.Tensor,
                user_features: torch.Tensor, 
                item_features: torch.Tensor,
                user_embeddings: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        
        user_ids = user_ids.to(self.device)
        item_ids = item_ids.to(self.device)
        user_features = user_features.to(self.device)
        item_features = item_features.to(self.device)
        
        # Get embeddings
        if user_embeddings is not None:
            user_emb_mlp = user_embeddings[0].repeat(len(user_ids), 1)
            user_emb_mf = user_embeddings[1].repeat(len(user_ids), 1)
        else:
            user_emb_mlp = self.user_embedding_mlp(user_ids)
            user_emb_mf = self.user_embedding_mf(user_ids)
        
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        item_emb_mf = self.item_embedding_mf(item_ids)
        
        # Transform features
        user_feat_transformed = self.user_feature_transform(user_features)
        item_feat_transformed = self.item_feature_transform(item_features)
        
        # Combine embeddings and features
        user_combined_mlp = torch.cat([user_emb_mlp, user_feat_transformed], dim=-1)
        item_combined_mlp = torch.cat([item_emb_mlp, item_feat_transformed], dim=-1)
        
        # MLP branch
        mlp_input = torch.cat([user_combined_mlp, item_combined_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Matrix Factorization branch
        mf_output = torch.mul(user_emb_mf, item_emb_mf)
        
        # Combine both branches
        final_input = torch.cat([mlp_output, mf_output], dim=-1)
        prediction = self.prediction_layer(final_input)
        
        return self.sigmoid(prediction).flatten()


class TrainingEarlyStopping:
    """Early stopping for model training"""
    
    def __init__(self, patience: int = 3, delta: float = 0, verbose: bool = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float, model: nn.Module):
        """Check early stopping condition"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.6f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
