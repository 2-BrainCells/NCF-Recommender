"""
Configuration file for Dyslexia Recommendation System
"""
import torch
from typing import List, Dict

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model hyperparameters (default values)
DEFAULT_CONFIG = {
    'embedding_dims': 256,
    'hidden_dims': [512, 1024],
    'dropout': 0.14,
    'learning_rate': 5.2e-05,
    'weight_decay': 1.98e-05,
    'batch_size': 128,
    'epochs': 10,
    'validation_split': 0.2,
    'early_stopping_patience': 3
}

# HPO configuration
HPO_CONFIG = {
    'n_trials': 100,
    'timeout': 3600,
    'patience': 10,
    'min_delta': 0.001
}

# Data processing configuration
DATA_CONFIG = {
    'max_missing_ratio': 0.35,
    'knn_neighbors': 5,
    'rating_scale': 5.0,
    'exponential_decay_factor': 0.6
}

# Item category mapping
CATEGORY_MAPPING = {
    'T1': 'Audio Book Tools', 'T2': 'Audio Book Tools',
    'T3': 'Color-Coded Text', 'T4': 'Assistive Writing Tools',
    'T5': 'Assistive Writing Tools', 'T6': 'Text Structuring Tools',
    'T7': 'Text Structuring Tools', 'T8': 'Pre-made Visual Aids',
    'T9': 'Pre-made Visual Aids', 'T10': 'Pre-made Visual Aids',
    'T11': 'Digital Books', 'T12': 'Digital Tutor',
    'T13': 'Visual Memory Aids', 'T14': 'Visual Memory Aids',
    'T15': 'Multimedia Lesson Recording', 'T16': 'Multimedia Lesson Recording',
    'S15': 'Multimedia Lesson Recording', 'T17': 'Supplementary Research',
    'S1': 'Personal Reader Support', 'S2': 'Self-made Study Aids',
    'S3': 'Self-made Study Aids', 'S4': 'Self-made Study Aids',
    'S5': 'Repetition Strategy', 'S6': 'Active Reading Markup',
    'S7': 'Active Reading Markup', 'S8': 'Group Study',
    'S9': 'Tutoring Support', 'S10': 'Peer Association',
    'S11': 'In-person Attendance', 'S12': 'Online Lessons',
    'S13': 'Classroom Support Aids', 'S14': 'Classroom Support Aids',
    'S16': 'Note Taking', 'S17': 'Lesson Planning',
    'S18': 'Assessment Adaptation', 'S19': 'Written Assessment',
    'S20': 'Oral Assessment', 'S21': 'Individual Assessment',
    'S22': 'Online Study Resources'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'dyslexia_recommendation.log'
}
