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

# Konfigürasyon modüllerini import et
from config.settings import DATABASE_URI, LOG_FILE, DEBUG_MODE, ALGORITHMS, DEEP_LEARNING
from config.logging import configure_logging

# Backend modüllerini import et
from backend.database.db_manager import DatabaseManager
from backend.engine.prediction_engine import PredictionEngine
from backend.engine.result_aggregator import ResultAggregator
from backend.engine.performance_tracker import PerformanceTracker
from backend.engine.weight_optimizer import WeightOptimizer

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
from backend.deep_learning.model_registry import ModelRegistry

# Konsol arayüzünü import et
from frontend.simple_ui import BaccaratConsoleUI

# Loglama ayarları
configure_logging(LOG_FILE, logging.DEBUG if DEBUG_MODE else logging.INFO, DEBUG_MODE)
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
    
    # Ağırlık optimizasyonu modülünü oluştur
    weight_optimizer = WeightOptimizer(db_manager, None)  # prediction_engine sonra ayarlanacak
    
    # Tahmin motorunu oluştur
    prediction_engine = PredictionEngine()
    prediction_engine.set_result_aggregator(result_aggregator)
    prediction_engine.set_db_manager(db_manager)
    
    # weight_optimizer'a prediction_engine'i ayarla
    weight_optimizer.prediction_engine = prediction_engine
    
    # Tüm algoritmaları kaydettir
    register_algorithms(prediction_engine, db_manager)
    
    # LSTM modelini oluştur ve yükle
    if DEEP_LEARNING['enabled']:
        initialize_lstm_model(prediction_engine, db_manager)
    
    logger.info("Sistem başlatma tamamlandı.")
    return db_manager, prediction_engine, performance_tracker, weight_optimizer

def register_algorithms(prediction_engine, db_manager):
    """Tüm algoritmaları tahmin motoruna kaydeder"""
    algorithms_config = ALGORITHMS
    registered_count = 0
    
    # PatternAnalysis
    if algorithms_config['pattern_analysis']['enabled']:
        prediction_engine.register_algorithm(
            PatternAnalysis(
                weight=algorithms_config['pattern_analysis']['weight'],
                db_manager=db_manager,
                min_samples=algorithms_config['pattern_analysis']['min_samples'],
                pattern_length=algorithms_config['pattern_analysis']['pattern_length']
            )
        )
        registered_count += 1
    
    # StatisticalModel
    if algorithms_config['statistical_model']['enabled']:
        prediction_engine.register_algorithm(
            StatisticalModel(
                weight=algorithms_config['statistical_model']['weight'],
                db_manager=db_manager
            )
        )
        registered_count += 1
    
    # SequenceAnalysis
    if algorithms_config['sequence_analysis']['enabled']:
        prediction_engine.register_algorithm(
            SequenceAnalysis(
                weight=algorithms_config['sequence_analysis']['weight'],
                db_manager=db_manager,
                sequence_length=algorithms_config['sequence_analysis']['sequence_length']
            )
        )
        registered_count += 1
    
    # BayesModel
    if algorithms_config['bayes_model']['enabled']:
        prediction_engine.register_algorithm(
            BayesModel(
                weight=algorithms_config['bayes_model']['weight'],
                db_manager=db_manager
            )
        )
        registered_count += 1
    
    # MarkovModel
    if algorithms_config['markov_model']['enabled']:
        prediction_engine.register_algorithm(
            MarkovModel(
                weight=algorithms_config['markov_model']['weight'],
                db_manager=db_manager,
                order=algorithms_config['markov_model']['order']
            )
        )
        registered_count += 1
    
    # CyclicalAnalysis
    if algorithms_config['cyclical_analysis']['enabled']:
        prediction_engine.register_algorithm(
            CyclicalAnalysis(
                weight=algorithms_config['cyclical_analysis']['weight'],
                db_manager=db_manager,
                cycle_lengths=algorithms_config['cyclical_analysis']['cycle_lengths']
            )
        )
        registered_count += 1
    
    # CorrelationModel
    if algorithms_config['correlation_model']['enabled']:
        prediction_engine.register_algorithm(
            CorrelationModel(
                weight=algorithms_config['correlation_model']['weight'],
                db_manager=db_manager
            )
        )
        registered_count += 1
    
    # MonteCarloSimulation
    if algorithms_config['monte_carlo']['enabled']:
        prediction_engine.register_algorithm(
            MonteCarloSimulation(
                weight=algorithms_config['monte_carlo']['weight'],
                db_manager=db_manager,
                simulations=algorithms_config['monte_carlo']['simulations']
            )
        )
        registered_count += 1
    
    # ClusteringModel
    if algorithms_config['clustering_model']['enabled']:
        prediction_engine.register_algorithm(
            ClusteringModel(
                weight=algorithms_config['clustering_model']['weight'],
                db_manager=db_manager,
                n_clusters=algorithms_config['clustering_model']['n_clusters']
            )
        )
        registered_count += 1
    
    # EntropyModel
    if algorithms_config['entropy_model']['enabled']:
        prediction_engine.register_algorithm(
            EntropyModel(
                weight=algorithms_config['entropy_model']['weight'],
                db_manager=db_manager
            )
        )
        registered_count += 1
    
    # RegressionModel
    if algorithms_config['regression_model']['enabled']:
        prediction_engine.register_algorithm(
            RegressionModel(
                weight=algorithms_config['regression_model']['weight'],
                db_manager=db_manager
            )
        )
        registered_count += 1
    
    # CombinationAnalysis (son olarak en iyi performans için)
    if algorithms_config['combination_analysis']['enabled']:
        combo = CombinationAnalysis(
            weight=algorithms_config['combination_analysis']['weight'],
            db_manager=db_manager
        )
        
        # Combination Analysis'e diğer algoritmaları ekle
        for algorithm in prediction_engine.algorithms:
            combo.add_algorithm(algorithm, algorithm.weight)
        
        prediction_engine.register_algorithm(combo)
        registered_count += 1
    
    logger.info(f"Toplam {registered_count} algoritma kaydedildi")

def initialize_lstm_model(prediction_engine, db_manager):
    """LSTM modelini oluşturur ve yükler"""
    try:
        lstm_config = DEEP_LEARNING
        
        lstm_model = LSTMModel(
            input_size=lstm_config['input_size'],
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            output_size=lstm_config['output_size']
        )
        
        # Modeller klasörünü kontrol et/oluştur
        models_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Model kayıt modülünü oluştur
        model_registry = ModelRegistry(db_manager, models_dir)
        
        # Aktif model varsa yükle
        active_model = db_manager.get_active_model("LSTM")
        model_loaded = False
        
        if active_model:
            try:
                file_path = active_model['file_path']
                if os.path.exists(file_path):
                    model_loaded = lstm_model.load_model(file_path)
                    logger.info(f"Önceden eğitilmiş LSTM modeli yüklendi: {file_path}")
            except Exception as e:
                logger.error(f"LSTM modeli yükleme hatası: {str(e)}")
        
        # Model yüklenemediyse yeni bir model oluştur
        if not model_loaded:
            lstm_model.build_model()
            logger.info("Yeni LSTM modeli oluşturuldu")
        
        # Modeli tahmin motoruna ekle
        prediction_engine.set_deep_learning_model(lstm_model)
        
        # Modeli yeniden eğitme kontrolü
        if lstm_config['auto_retrain']:
            last_n_records = db_manager.get_last_n_results(1000)
            if len(last_n_records) >= 200:  # En az 200 kayıt varsa eğitime başla
                logger.info("Yeterli veri bulundu, LSTM modeli eğitiliyor...")
                
                model_trainer = ModelTrainer(lstm_model, db_manager)
                model_trainer.train_model(
                    epochs=lstm_config['epochs'],
                    batch_size=lstm_config['batch_size']
                )
    except Exception as e:
        logger.error(f"LSTM modeli başlatma hatası: {str(e)}")
        logger.info("LSTM modeli başlatılamadı, sistem LSTM olmadan devam edecek")

def main():
    """Ana uygulama başlangıç noktası"""
    parser = argparse.ArgumentParser(description='Baccarat Tahmin Sistemi')
    parser.add_argument('--reset-db', action='store_true', help='Veritabanını sıfırla')
    parser.add_argument('--retrain', action='store_true', help='LSTM modelini yeniden eğit')
    parser.add_argument('--optimize', action='store_true', help='Algoritma ağırlıklarını optimize et')
    parser.add_argument('--api', action='store_true', help='API sunucusunu başlat')
    parser.add_argument('--no-ui', action='store_true', help='Kullanıcı arayüzünü başlatma')
    args = parser.parse_args()
    
    # Veritabanını sıfırla
    if args.reset_db:
        db_manager = DatabaseManager(DATABASE_URI)
        db_manager.reset_database()
        logger.info("Veritabanı sıfırlandı.")
        return
    
    # Sistemi başlat
    db_manager, prediction_engine, performance_tracker, weight_optimizer = initialize_system()
    
    # LSTM modelini yeniden eğit
    if args.retrain:
        logger.info("LSTM modeli yeniden eğitiliyor...")
        model_trainer = ModelTrainer(prediction_engine.deep_learning_model, db_manager)
        model_trainer.train_model(epochs=DEEP_LEARNING['epochs'], batch_size=DEEP_LEARNING['batch_size'], force=True)
    
    # Algoritma ağırlıklarını optimize et
    if args.optimize:
        logger.info("Algoritma ağırlıkları optimize ediliyor...")
        weight_optimizer.optimize_weights()
    
    # API sunucusunu başlat
    if args.api:
        try:
            import uvicorn
            from backend.api.routes import create_app
            
            logger.info("API sunucusu başlatılıyor...")
            app = create_app(prediction_engine, performance_tracker, weight_optimizer, db_manager)
            uvicorn.run(app, host="0.0.0.0", port=5000)
            return
        except ImportError:
            logger.error("API sunucusu başlatılamadı. 'uvicorn' ve 'fastapi' paketlerini yükleyin.")
            logger.info("pip install uvicorn fastapi")
    
    # Kullanıcı arayüzünü başlat (console UI)
    if not args.no_ui:
        logger.info("Konsol kullanıcı arayüzü başlatılıyor...")
        ui = BaccaratConsoleUI()
        ui.db_manager = db_manager
        ui.prediction_engine = prediction_engine
        ui.performance_tracker = performance_tracker
        ui.run()
    else:
        logger.info("Baccarat Tahmin Sistemi çalışmaya hazır.")
        logger.info("UI olmadan çalışıyor. Çıkmak için Ctrl+C tuşlarına basın.")
        try:
            # Sonsuz döngü - UI olmadan çalıştırılırsa bekle
            while True:
                pass
        except KeyboardInterrupt:
            logger.info("Program sonlandırıldı.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Program hata ile sonlandı: {str(e)}")
        sys.exit(1)