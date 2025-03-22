#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baccarat Tahmin Sistemi - Ana Uygulama
---------------------------------------
14 farklı algoritma ve derin öğrenme entegrasyonu içeren baccarat tahmin yazılımı.
Her oyun sonucu veritabanına kaydedilir ve sistem sürekli öğrenmeye devam eder.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Backend modüllerini import et
from backend.database.db_manager import DatabaseManager
from backend.engine.prediction_engine import PredictionEngine
from backend.engine.result_aggregator import ResultAggregator
from backend.engine.performance_tracker import PerformanceTracker

# Algoritmaları import et
from backend.algorithms.pattern_analysis import PatternAnalysis
from backend.algorithms.statistical import StatisticalModel
from backend.algorithms.sequence import SequenceAnalysis
from backend.algorithms.bayes import BayesModel
from backend.algorithms.combination import CombinationAnalysis
from backend.algorithms.markov import MarkovModel
from backend.algorithms.cyclical import CyclicalAnalysis
from backend.algorithms.correlation import CorrelationModel
from backend.algorithms.monte_carlo import MonteCarloSimulation
from backend.algorithms.clustering import ClusteringModel
from backend.algorithms.time_series import TimeSeriesModel
from backend.algorithms.entropy import EntropyModel
from backend.algorithms.regression import RegressionModel

# Derin öğrenme modelini import et
from backend.deep_learning.lstm_model import LSTMModel
from backend.deep_learning.training import ModelTrainer

# Konfigürasyon
from config.settings import DATABASE_URI, LOG_FILE, DEBUG_MODE

# Loglama ayarları
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def initialize_system():
    """Sistemi başlatır ve bileşenleri ayarlar"""
    logger.info("Baccarat Tahmin Sistemi başlatılıyor...")
    
    # Veritabanı bağlantısını oluştur
    db_manager = DatabaseManager(DATABASE_URI)
    db_manager.connect()
    
    # Sonuç birleştiriciyi oluştur
    result_aggregator = ResultAggregator()
    
    # Performans izleyiciyi oluştur
    performance_tracker = PerformanceTracker(db_manager)
    
    # Tahmin motorunu oluştur
    prediction_engine = PredictionEngine()
    prediction_engine.set_result_aggregator(result_aggregator)
    
    # Tüm algoritmaları kaydettir
    algorithms = [
        PatternAnalysis(weight=1.0, db_manager=db_manager),
        StatisticalModel(weight=1.0, db_manager=db_manager),
        SequenceAnalysis(weight=1.0, db_manager=db_manager),
        BayesModel(weight=1.0, db_manager=db_manager),
        CombinationAnalysis(weight=1.0, db_manager=db_manager),
        MarkovModel(weight=1.0, db_manager=db_manager),
        CyclicalAnalysis(weight=1.0, db_manager=db_manager),
        CorrelationModel(weight=1.0, db_manager=db_manager),
        MonteCarloSimulation(weight=1.0, db_manager=db_manager),
        ClusteringModel(weight=1.0, db_manager=db_manager),
        TimeSeriesModel(weight=1.0, db_manager=db_manager),
        EntropyModel(weight=1.0, db_manager=db_manager),
        RegressionModel(weight=1.0, db_manager=db_manager),
    ]
    
    for algorithm in algorithms:
        prediction_engine.register_algorithm(algorithm)
    
    # LSTM modelini oluştur ve yükle
    lstm_model = LSTMModel(
        input_size=10,  # Son 10 oyun sonucu
        hidden_size=64,  # Gizli katman boyutu
        num_layers=2,   # LSTM katman sayısı
        output_size=3   # P/B/T için 3 çıktı
    )
    
    # Daha önce eğitilmiş model varsa yükle
    model_path = os.path.join('models', 'lstm_latest.h5')
    if os.path.exists(model_path):
        lstm_model.load_model(model_path)
        logger.info("Önceden eğitilmiş LSTM modeli yüklendi.")
    else:
        logger.info("Önceden eğitilmiş LSTM modeli bulunamadı, yeni model oluşturuldu.")
        lstm_model.build_model()
    
    # Derin öğrenme modelini tahmin motoruna ekle
    prediction_engine.set_deep_learning_model(lstm_model)
    
    # Modeli yeniden eğitme kontrolü
    last_n_records = db_manager.get_last_n_results(1000)
    if len(last_n_records) >= 200:  # En az 200 kayıt varsa eğitime başla
        logger.info("Yeterli veri bulundu, LSTM modeli eğitiliyor...")
        model_trainer = ModelTrainer(lstm_model, db_manager)
        model_trainer.train_model(epochs=10, batch_size=32)
    
    logger.info("Sistem başlatma tamamlandı.")
    return db_manager, prediction_engine, performance_tracker

def main():
    """Ana uygulama başlangıç noktası"""
    parser = argparse.ArgumentParser(description='Baccarat Tahmin Sistemi')
    parser.add_argument('--reset-db', action='store_true', help='Veritabanını sıfırla')
    parser.add_argument('--retrain', action='store_true', help='LSTM modelini yeniden eğit')
    parser.add_argument('--optimize', action='store_true', help='Algoritma ağırlıklarını optimize et')
    args = parser.parse_args()
    
    # Veritabanını sıfırla
    if args.reset_db:
        db_manager = DatabaseManager(DATABASE_URI)
        db_manager.reset_database()
        logger.info("Veritabanı sıfırlandı.")
        return
    
    # Sistemi başlat
    db_manager, prediction_engine, performance_tracker = initialize_system()
    
    # LSTM modelini yeniden eğit
    if args.retrain:
        logger.info("LSTM modeli yeniden eğitiliyor...")
        model_trainer = ModelTrainer(prediction_engine.deep_learning_model, db_manager)
        model_trainer.train_model(epochs=20, batch_size=32, force=True)
    
    # Algoritma ağırlıklarını optimize et
    if args.optimize:
        logger.info("Algoritma ağırlıkları optimize ediliyor...")
        performance_tracker.optimize_weights()
    
    logger.info("Baccarat Tahmin Sistemi çalışmaya hazır.")
    
    # Burada arayüz başlatılabilir veya API servisi çalıştırılabilir
    # Örneğin: start_user_interface(db_manager, prediction_engine)
    
    logger.info("Program sonlandı.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Program hata ile sonlandı: {str(e)}")
        sys.exit(1)