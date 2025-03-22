#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performans İzleyici
------------------
Algoritmaların performansını izler ve ağırlıklarını optimize eder.
"""

import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Algoritmaların performansını izleyen ve optimize eden sınıf
    
    Algoritmalar doğru tahmin yaptıkça ağırlıkları artırılır,
    yanlış tahmin yaptıkça azaltılır.
    """
    
    def __init__(self, db_manager):
        """
        PerformanceTracker sınıfını başlatır
        
        Args:
            db_manager: Veritabanı yönetici nesnesi
        """
        self.db_manager = db_manager
        self.optimization_interval = 50  # Kaç sonuçta bir optimizasyon yapılacak
        self.min_weight = 0.1  # Minimum algoritma ağırlığı
        self.max_weight = 10.0  # Maksimum algoritma ağırlığı
        
        logger.info("Performans izleyici başlatıldı")
    
    def evaluate_predictions(self, predictions, actual_result):
        """
        Tahminlerin doğruluğunu değerlendirir
        
        Args:
            predictions (list): Algoritma tahminlerinin listesi
            actual_result (str): Gerçekleşen sonuç (P/B/T)
        
        Returns:
            dict: Değerlendirme sonuçları
        """
        if not predictions:
            logger.warning("Değerlendirilecek tahmin yok")
            return {}
        
        # Her algoritma için doğru/yanlış tahminleri belirle
        evaluation = {}
        
        for pred in predictions:
            algorithm_name = pred['algorithm']
            predicted_result = pred['prediction']
            confidence = pred['confidence']
            
            is_correct = predicted_result == actual_result
            
            evaluation[algorithm_name] = {
                'predicted': predicted_result,
                'actual': actual_result,
                'is_correct': is_correct,
                'confidence': confidence
            }
            
            logger.debug(f"{algorithm_name}: Tahmin={predicted_result}, Gerçek={actual_result}, Doğru={is_correct}")
        
        return evaluation
    
    def update_algorithm_metrics(self, algorithm_id, is_correct, confidence):
        """
        Algoritma metriklerini günceller
        
        Args:
            algorithm_id (int): Algoritmanın ID'si
            is_correct (bool): Tahmin doğru mu
            confidence (float): Tahmin güven skoru
        
        Returns:
            bool: İşlem başarılı ise True, değilse False
        """
        try:
            # Algoritma bilgilerini getir
            algorithm_info = self.db_manager.get_algorithm_by_id(algorithm_id)
            
            if not algorithm_info:
                logger.warning(f"Algoritma bulunamadı: ID={algorithm_id}")
                return False
            
            # Mevcut metrikleri güncelle
            total_predictions = algorithm_info['total_predictions'] + 1
            correct_predictions = algorithm_info['correct_predictions'] + (1 if is_correct else 0)
            accuracy = correct_predictions / total_predictions
            
            # Veritabanını güncelle
            self.db_manager.update_algorithm_performance(
                algorithm_id=algorithm_id,
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                accuracy=accuracy
            )
            
            logger.debug(f"Algoritma metrikleri güncellendi: ID={algorithm_id}, Doğru={is_correct}, Doğruluk={accuracy:.4f}")
            return True
        except Exception as e:
            logger.error(f"Algoritma metrikleri güncelleme hatası: {str(e)}")
            return False
    
    def optimize_weights(self):
        """
        Algoritma ağırlıklarını performanslarına göre optimize eder
        
        Returns:
            dict: Güncellenen ağırlıklar
        """
        try:
            # Tüm algoritmaları getir
            algorithms = self.db_manager.get_all_algorithms()
            
            if not algorithms:
                logger.warning("Optimize edilecek algoritma bulunamadı")
                return {}
            
            # Son 100 sonucu getir
            last_results = self.db_manager.get_last_n_results(100)
            
            if len(last_results) < 10:
                logger.warning(f"Yetersiz sonuç sayısı: {len(last_results)} < 10")
                return {}
            
            updated_weights = {}
            
            # Her algoritma için performans analizi yap
            for algorithm in algorithms:
                algorithm_id = algorithm['id']
                algorithm_name = algorithm['name']
                current_weight = algorithm['weight']
                current_accuracy = algorithm['current_accuracy']
                
                # Son sonuçlarda algoritmanın performansını değerlendir
                performance = self.db_manager.get_algorithm_performance_in_timeframe(
                    algorithm_id=algorithm_id,
                    days=7  # Son 7 günlük performans
                )
                
                if not performance or performance['total_predictions'] == 0:
                    logger.warning(f"Algoritma için performans verisi bulunamadı: {algorithm_name}")
                    continue
                
                recent_accuracy = performance['accuracy']
                
                # Ağırlık ayarlaması yap
                if recent_accuracy > current_accuracy:
                    # Performans iyileşiyor, ağırlığı artır
                    new_weight = min(current_weight * 1.2, self.max_weight)
                else:
                    # Performans kötüleşiyor, ağırlığı azalt
                    new_weight = max(current_weight * 0.9, self.min_weight)
                
                # Ağırlığı güncelle
                self.db_manager.update_algorithm_weight(algorithm_id, new_weight)
                
                updated_weights[algorithm_name] = {
                    'old_weight': current_weight,
                    'new_weight': new_weight,
                    'current_accuracy': current_accuracy,
                    'recent_accuracy': recent_accuracy
                }
                
                logger.info(f"Algoritma ağırlığı güncellendi: {algorithm_name}, {current_weight:.2f} -> {new_weight:.2f}")
            
            return updated_weights
        except Exception as e:
            logger.error(f"Ağırlık optimizasyonu hatası: {str(e)}")
            return {}
    
    def check_model_retraining(self, threshold_days=7, min_accuracy_improvement=0.02):
        """
        Derin öğrenme modelinin yeniden eğitim gereksinimini kontrol eder
        
        Args:
            threshold_days (int): Yeniden eğitim için gün eşiği
            min_accuracy_improvement (float): Minimum doğruluk iyileştirme eşiği
        
        Returns:
            bool: Yeniden eğitim gerekli ise True, değilse False
        """
        try:
            # Aktif modeli getir
            active_model = self.db_manager.get_active_model("LSTM")
            
            if not active_model:
                logger.info("Aktif LSTM modeli bulunamadı, yeniden eğitim gerekli")
                return True
            
            # Model yaratılma zamanını kontrol et
            created_at = datetime.fromisoformat(active_model['created_at'])
            days_since_creation = (datetime.now() - created_at).days
            
            # Belirli bir süreden eski ise yeniden eğit
            if days_since_creation >= threshold_days:
                logger.info(f"Model {days_since_creation} gündür güncellenmedi, yeniden eğitim gerekli")
                return True
            
            # Son sonuçları getir
            last_results = self.db_manager.get_last_n_results(200)
            
            if len(last_results) < 100:
                logger.info("Yeterli yeni veri yok, yeniden eğitim gerekli değil")
                return False
            
            # Model performansını kontrol et
            model_accuracy = active_model['accuracy']
            
            # Son tahminlerde LSTM'in performansını değerlendir
            lstm_performance = self.db_manager.get_algorithm_performance_by_name(
                algorithm_name="LSTM",
                days=3  # Son 3 günlük performans
            )
            
            if not lstm_performance or lstm_performance['total_predictions'] < 20:
                logger.info("LSTM için yeterli performans verisi yok")
                return False
            
            recent_accuracy = lstm_performance['accuracy']
            
            # Eğer son performans modelin kayıtlı doğruluğundan belirli bir miktar düşükse yeniden eğit
            if model_accuracy - recent_accuracy > min_accuracy_improvement:
                logger.info(f"LSTM performansı düştü: {model_accuracy:.4f} -> {recent_accuracy:.4f}, yeniden eğitim gerekli")
                return True
            
            logger.info("LSTM yeniden eğitimi gerekli değil")
            return False
        except Exception as e:
            logger.error(f"Model yeniden eğitim kontrolü hatası: {str(e)}")
            return False
    
    def get_performance_stats(self, days=30):
        """
        Performans istatistiklerini getirir
        
        Args:
            days (int): Kaç günlük istatistik getirileceği
        
        Returns:
            dict: Performans istatistikleri
        """
        try:
            # Tüm algoritmaları getir
            algorithms = self.db_manager.get_all_algorithms()
            
            if not algorithms:
                return {}
            
            # Son X günlük performans verilerini getir
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            stats = {
                'overall': {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'accuracy': 0.0
                },
                'algorithms': {}
            }
            
            # Her algoritma için performans verilerini getir
            for algorithm in algorithms:
                algorithm_id = algorithm['id']
                algorithm_name = algorithm['name']
                
                performance = self.db_manager.get_algorithm_performance_since_date(
                    algorithm_id=algorithm_id,
                    start_date=start_date
                )
                
                if performance:
                    stats['algorithms'][algorithm_name] = performance
                    
                    # Genel istatistikleri güncelle
                    stats['overall']['total_predictions'] += performance['total_predictions']
                    stats['overall']['correct_predictions'] += performance['correct_predictions']
            
            # Genel doğruluk oranını hesapla
            if stats['overall']['total_predictions'] > 0:
                stats['overall']['accuracy'] = stats['overall']['correct_predictions'] / stats['overall']['total_predictions']
            
            return stats
        except Exception as e:
            logger.error(f"Performans istatistikleri getirme hatası: {str(e)}")
            return {}