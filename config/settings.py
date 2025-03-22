#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baccarat Tahmin Sistemi Yapılandırma Ayarları
-------------------------------------------
"""

import os
import logging
from datetime import datetime

# Temel dizinler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Dizinlerin varlığını kontrol et ve oluştur
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Veritabanı ayarları
DATABASE_URI = os.path.join(DATA_DIR, 'baccarat.db')

# Loglama ayarları
LOG_FILE = os.path.join(LOGS_DIR, f'baccarat_{datetime.now().strftime("%Y%m%d")}.log')
LOG_LEVEL = logging.INFO
DEBUG_MODE = True

# Algoritma ayarları
ALGORITHMS = {
    'pattern_analysis': {
        'enabled': True,
        'weight': 1.0,
        'min_samples': 5,
        'pattern_length': 3
    },
    'statistical_model': {
        'enabled': True,
        'weight': 1.0
    },
    'sequence_analysis': {
        'enabled': True,
        'weight': 1.0,
        'sequence_length': 5
    },
    'bayes_model': {
        'enabled': True,
        'weight': 1.0
    },
    'combination_analysis': {
        'enabled': True,
        'weight': 1.0
    },
    'markov_model': {
        'enabled': True,
        'weight': 1.0,
        'order': 2
    },
    'cyclical_analysis': {
        'enabled': True,
        'weight': 1.0,
        'cycle_lengths': [5, 7, 9, 11]
    },
    'correlation_model': {
        'enabled': True,
        'weight': 1.0
    },
    'monte_carlo': {
        'enabled': True,
        'weight': 1.0,
        'simulations': 1000
    },
    'clustering_model': {
        'enabled': True,
        'weight': 1.0,
        'n_clusters': 3
    },
    'time_series': {
        'enabled': True,
        'weight': 1.0,
        'window_size': 10
    },
    'entropy_model': {
        'enabled': True,
        'weight': 1.0
    },
    'regression_model': {
        'enabled': True,
        'weight': 1.0
    }
}

# Derin öğrenme ayarları
DEEP_LEARNING = {
    'enabled': True,
    'weight': 1.5,
    'input_size': 10,
    'hidden_size': 64,
    'num_layers': 2,
    'output_size': 3,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'train_test_split': 0.2,
    'auto_retrain': True,
    'retrain_threshold_days': 7,
    'min_accuracy_improvement': 0.02
}

# Sonuç birleştirme ayarları
AGGREGATION = {
    'strategy': 'weighted_voting',  # weighted_voting, confidence_adjusted, accuracy_based
    'min_confidence': 0.4
}

# Uygulama ayarları
APP_SETTINGS = {
    'session_lifetime': 86400,  # 1 gün (saniye cinsinden)
    'auto_optimization_interval': 50,  # Her 50 sonuçta bir otomatik optimizasyon
    'min_data_for_prediction': 10  # Tahmin için gerekli minimum veri sayısı
}

# API ayarları
API_SETTINGS = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': DEBUG_MODE,
    'secret_key': 'supersecretkey123',  # Üretim ortamında değiştirin!
    'token_expiration': 3600  # 1 saat (saniye cinsinden)
}