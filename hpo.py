import optuna
import pickle
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn

from model import NeuralCollaborativeFiltering, TrainingEarlyStopping
from utils import HiddenPrints, train_model_epoch
from config import HPO_CONFIG

class EarlyStoppingCallback:
    """Optuna callback implementing early stopping based on study performance to prevent excessive optimization time."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """Initialize early stopping with patience threshold and minimum improvement delta."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        """Evaluate study progress and trigger early stopping when improvement stagnates."""
        if study.best_value < self.best_value - self.min_delta:
            self.best_value = study.best_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()

class HyperparameterOptimizer:
    """Automated hyperparameter optimization system using Optuna framework for neural collaborative filtering model tuning."""
    
    def __init__(self, data_processor, device: str = 'cpu'):
        """Initialize optimizer with data processor and computing device configuration."""
        self.data_processor = data_processor
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.study = None
        self.best_params = None

    def objective_function(self, trial: optuna.Trial) -> float:
        """Define optimization objective function that evaluates model performance with suggested hyperparameters."""
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        embedding_dims = trial.suggest_categorical('embedding_dims', [16, 32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        user_feature_dim = self.data_processor.user_data.shape[1]
        item_feature_dim = self.data_processor.item_data.shape[1]
        num_users = self.data_processor.num_users
        num_items = self.data_processor.num_items

        # Add validation
        from utils import validate_embedding_indices
        validate_embedding_indices(
            self.data_processor.ratings_data,
            num_users,
            num_items
        )

        train_data, val_data, _ = self.data_processor.split_data(
            self.data_processor.user_data,
            self.data_processor.item_data,
            self.data_processor.ratings_data,
            validation_split=0.2
        )


        model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim,
            embedding_dims=embedding_dims,
            hidden_dims=[],
            dropout=dropout,
            device=self.device,
            trial=trial
        )

        model.to(self.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        early_stopping = TrainingEarlyStopping(patience=3, delta=0.0002)

        with HiddenPrints():
            history = train_model_epoch(
                model, train_data, val_data,
                epochs=20, batch_size=batch_size,
                optimizer=optimizer, criterion=criterion,
                early_stopping=early_stopping,
                device=self.device
            )

        return sum(history['val_loss']) / len(history['val_loss'])

    def run_optimization(self,
                        n_trials: int = None,
                        timeout: int = None,
                        save_study: bool = True,
                        study_name: str = 'dyslexia_ncf_optimization') -> optuna.Study:
        """Execute hyperparameter optimization process with configurable trial limits and time constraints."""
        n_trials = n_trials or HPO_CONFIG['n_trials']
        timeout = timeout or HPO_CONFIG['timeout']

        self.logger.info("Starting hyperparameter optimization...")

        self.study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        early_stopping_callback = EarlyStoppingCallback(
            patience=HPO_CONFIG['patience'],
            min_delta=HPO_CONFIG['min_delta']
        )

        try:
            self.study.optimize(
                self.objective_function,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[early_stopping_callback]
            )

            self.best_params = self.study.best_params

            self.logger.info("Hyperparameter optimization completed!")
            self.logger.info(f"Best trial value: {self.study.best_value:.6f}")
            self.logger.info(f"Best parameters: {self.best_params}")

            if save_study:
                study_path = f'{study_name}_study.pkl'
                with open(study_path, 'wb') as f:
                    pickle.dump(self.study, f)
                self.logger.info(f"Study saved to {study_path}")

        except Exception as e:
            self.logger.error(f"Error during HPO: {str(e)}")
            raise

        return self.study

    def get_best_config(self) -> Dict:
        """Extract optimal hyperparameter configuration from completed optimization study."""
        if self.best_params is None:
            raise ValueError("No optimization results available")

        hidden_dims = []
        layer_keys = [k for k in self.best_params.keys() if k.startswith('hidden_dim_layer_')]
        layer_keys.sort(key=lambda x: int(x.split('_')[-1]))
        hidden_dims = [self.best_params[k] for k in layer_keys]

        return {
            'embedding_dims': self.best_params.get('embedding_dims', 256),
            'hidden_dims': hidden_dims,
            'dropout': self.best_params.get('dropout', 0.14),
            'learning_rate': self.best_params.get('lr', 5.2e-05),
            'weight_decay': self.best_params.get('weight_decay', 1.98e-05),
            'batch_size': self.best_params.get('batch_size', 128)
        }
