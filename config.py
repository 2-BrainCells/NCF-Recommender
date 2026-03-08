import torch
from typing import List, Dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SYSTEM_CONFIG = {
    'embedding_dims': 16,            
    'hidden_dims': [64, 32],         
    'dropout': 0.2,
    'learning_rate': 1e-3,           
    'weight_decay': 1e-5,
    'batch_size': 64,             
    'epochs': 20,
    'validation_split': 0.2,
    'use_p_features': False,          # ABLATION SWITCH: Set to False to disable P1-P12 features
    'early_stopping_patience': 3,
    'top_k_metrics': 5,      # Calculate HitRate/NDCG at this K
    'hit_threshold': 0.5,     # >0.5 probability is considered a hit (BCE Loss)
    'recommendation_k': 5,    # How many items to show the user in UI
    'diversity_cap': 2,       # Max items from the same category allowed
    'preference_boost': 0.5   # Rating boost applied to explicitly preferred categories
}

HPO_CONFIG = {
    'n_trials': 50,
    'timeout': 1800,
    'patience': 5,
    'min_delta': 0.001
}

DATA_CONFIG = {
    'rating_scale': 5.0,
    'exponential_decay_factor': 0.6
}

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

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'dyslexia_recommendation.log'
}