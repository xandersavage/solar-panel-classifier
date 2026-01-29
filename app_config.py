"""
Configuration file for Solar Panel Classifier
Contains all configuration settings and constants
"""

import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models/solar_panel_classifier_efficientnetb0.h5') # Updated to root based on notebook save
DATA_DIR = os.path.join(BASE_DIR, 'data')
HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.json')

# ==================== GRID METRICS ====================
# Metrics from the fine-tuned model evaluation
MODEL_METRICS = {
    "Accuracy": "81.4%",
    "Loss": "0.61",
    "F1_Scores": {
        "Physical-Damage": "0.94",
        "Snow-Covered": "0.95",
        "Bird-drop": "0.83",
        "Electrical-damage": "0.77",
        "Clean": "0.74",
        "Dusty": "0.73"
    },
    "Key_Insight": "High reliability in detecting critical defects (Physical Damage & Snow)."
}

# ==================== MODEL CONFIGURATION ====================
IMG_SIZE = (224, 224)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32

# ==================== CLASSES ====================
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

# ==================== CLASS INFORMATION ====================
CLASS_DESCRIPTIONS = {
    "Bird-drop": "Bird droppings obstruct sunlight and reduce panel efficiency",
    "Clean": "Panel is in optimal condition with no visible defects",
    "Dusty": "Dust accumulation reduces light absorption and efficiency",
    "Electrical-damage": "Electrical malfunction requiring immediate professional attention",
    "Physical-Damage": "Physical damage to panel structure, may include cracks or breaks",
    "Snow-Covered": "Snow coverage blocking sunlight and reducing energy production"
}

SEVERITY_LEVELS = {
    "Bird-drop": "Low",
    "Clean": "None",
    "Dusty": "Low",
    "Electrical-damage": "High",
    "Physical-Damage": "High",
    "Snow-Covered": "Medium"
}

RECOMMENDED_ACTIONS = {
    "Bird-drop": "Clean panel with water and soft brush. Consider installing bird deterrents.",
    "Clean": "No action needed. Continue regular monitoring and maintenance schedule.",
    "Dusty": "Clean panel with water spray. Schedule regular cleaning based on local conditions.",
    "Electrical-damage": "⚠️ URGENT: Contact qualified electrician immediately. Do not operate the system.",
    "Physical-Damage": "⚠️ URGENT: Inspect for cracks and structural integrity. Panel replacement may be required.",
    "Snow-Covered": "Remove snow carefully using appropriate tools or wait for natural melting. Do not use hot water."
}

# ==================== UI COLORS ====================
SEVERITY_COLORS = {
    'None': '#10b981',    # Green
    'Low': '#f59e0b',     # Yellow/Orange
    'Medium': '#f97316',  # Orange
    'High': '#ef4444'     # Red
}

# ==================== CONFIDENCE THRESHOLDS ====================
CONFIDENCE_THRESHOLDS = {
    'very_high': 0.90,
    'high': 0.75,
    'moderate': 0.50,
    'low': 0.0
}

# ==================== APP SETTINGS ====================
APP_TITLE = "Solar Panel AI Inspector"
APP_ICON = "☀️"
PAGE_LAYOUT = "wide"
AUTHOR = "Alexander Olomukoro"
VERSION = "1.0.0"

# ==================== TRAINING CONFIGURATION ====================
# These are used in the training notebook
TRAINING_CONFIG = {
    'epochs': 25,
    'learning_rate': 1e-4,
    'validation_split': 0.2,
    'seed': 42,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5,
}

# ==================== DATA AUGMENTATION ====================
AUGMENTATION_CONFIG = {
    'rotation_range': 0.2,
    'zoom_range': 0.2,
    'contrast_range': 0.2,
    'brightness_range': 0.2,
    'horizontal_flip': True,
}
