#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algoritma Ağırlık Optimizasyon Modülü
-----------------------------------
Algoritma ağırlıklarını performansa göre otomatik olarak optimize eder.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class WeightOptimizer:
    """
    Algoritma ağırlıklarını optimize eden sınıf
    
    Farklı optimizasyon stratejileri uygulayarak algoritma ağırlıklarını
    performanslarına göre ayarlar.
    """
    
    def __init__(self, db_manager, prediction_engine):
        """
        WeightOptimizer sınıfını başlatır
        
        Args:
            db_manager: Veritabanı yönetici nesnesi
            prediction_engine: Tahmin motoru nesnesi
        """
        self.db_manager = db_manager
        self.prediction_engine = prediction_engine
        self.min_weight = 0.1  # Minimum algoritma ağırlığı
        self.max_weight = 5.0  # Maksimum algoritma ağırlığı
        self.default_weight = 1.0  # Varsayılan ağırlık
        
        logger.info("Ağırlık optimizasyonu modülü başlatıldı")
    
    def optimize_weights(self, strategy="performance", days=7):
        """
        Algoritma ağırlıklarını optimize eder
        
        Args:
            strategy (str): Optimizasyon stratejisi
                - performance: Performansa dayalı ağırlıklandırma
                - adaptive: Uyarlanabilir ağırlıklandırma
                - balanced: Dengeli ağırlıklandırma
                - random_search: Rastgele arama
            days (int): Performans analizinde kullanılacak gün sayısı
        
        Returns:
            dict: Optimizasyon sonuçları
        """
        logger.info(f"Ağırlık optimizasyonu başlatıldı: Strateji={strategy}, Gün={days}")
        
        # Strateji seçimi
        if strategy == "performance":
            return self._performance_based_optimization(days)
        elif strategy == "adaptive":
            return self._adaptive_optimization(days)
        elif strategy == "balanced":
            return self._balanced_optimization(days)
        elif strategy == "random_search":
            return self._random_search_optimization(days)
        else:
            logger.warning(f"Geçersiz optimizasyon stratejisi: {strategy}, performance stratejisi kullanılıyor")
            return self._performance_based_optimization(days)
    
    def _performance_based_optimization(self, days=7):
        """
        Performansa dayalı ağırlık optimizasyonu
        
        Args:
            days (int): Performans analizinde kullanılacak gün sayısı
        
        Returns:
            dict: Optimizasyon sonuçları
        """
        try:
            results = {}
            
            # Tüm algoritmaları getir
            algorithms = self.prediction_engine.algorithms
            
            for algorithm in algorithms:
                algorithm_name = algorithm.name
                current_weight = algorithm.weight
                
                # Algoritma ID'sini bul
                algorithm_info = self.db_manager.get_algorithm_by_name(algorithm_name)
                
                if not algorithm_info:
                    logger.warning(f"Algoritma veritabanında bulunamadı: {algorithm_name}")
                    continue
                
                algorithm_id = algorithm_info['id']
                
                # Son N günlük performansı getir
                start_date = (datetime.now() - timedelta(days=days)).isoformat()
                performance = self.db_manager.get_algorithm_performance_since_date(algorithm_id, start_date)
                
                if not performance or performance['total_predictions'] < 10:
                    logger.warning(f"Yeterli performans verisi yok: {algorithm_name}")
                    results[algorithm_name] = {
                        'old_weight': current_weight,
                        'new_weight': current_weight,
                        'accuracy': algorithm.accuracy,
                        'message': "Yeterli veri yok"
                    }
                    continue
                
                # Algoritma doğruluğunu getir
                accuracy = performance['accuracy']
                
                # Mevcut ağırlığa göre ayarla
                if accuracy >= 0.7:  # Çok iyi performans
                    new_weight = min(current_weight * 1.5, self.max_weight)
                elif accuracy >= 0.6:  # İyi performans
                    new_weight = min(current_weight * 1.2, self.max_weight)
                elif accuracy >= 0.5:  # Ortalama performans
                    new_weight = current_weight  # Değişiklik yok
                elif accuracy >= 0.4:  # Kötü performans
                    new_weight = max(current_weight * 0.8, self.min_weight)
                else:  # Çok kötü performans
                    new_weight = max(current_weight * 0.5, self.min_weight)
                
                # Ağırlığı güncelle
                algorithm.set_weight(new_weight)
                
                # Veritabanını güncelle
                self.db_manager.update_algorithm_weight(algorithm_id, new_weight)
                
                # Sonuçları kaydet
                results[algorithm_name] = {
                    'old_weight': current_weight,
                    'new_weight': new_weight,
                    'accuracy': accuracy,
                    'total_predictions': performance['total_predictions'],
                    'correct_predictions': performance['correct_predictions']
                }
                
                logger.info(f"Algoritma ağırlığı güncellendi: {algorithm_name}, {current_weight:.2f} -> {new_weight:.2f}")
            
            return results
        except Exception as e:
            logger.error(f"Performans tabanlı optimizasyon hatası: {str(e)}")
            return {}
    
    def _adaptive_optimization(self, days=7):
        """
        Uyarlanabilir ağırlık optimizasyonu
        
        Algoritmaların göreceli performansına ve son eğilimlere göre ağırlıkları ayarlar.
        
        Args:
            days (int): Performans analizinde kullanılacak gün sayısı
        
        Returns:
            dict: Optimizasyon sonuçları
        """
        try:
            results = {}
            
            # Tüm algoritmaları getir
            algorithms = self.prediction_engine.algorithms
            performance_data = []
            
            # Her algoritmanın son performansını getir
            for algorithm in algorithms:
                algorithm_name = algorithm.name
                current_weight = algorithm.weight
                
                # Algoritma ID'sini bul
                algorithm_info = self.db_manager.get_algorithm_by_name(algorithm_name)
                
                if not algorithm_info:
                    logger.warning(f"Algoritma veritabanında bulunamadı: {algorithm_name}")
                    continue
                
                algorithm_id = algorithm_info['id']
                
                # Son N günlük performansı getir
                start_date = (datetime.now() - timedelta(days=days)).isoformat()
                performance = self.db_manager.get_algorithm_performance_since_date(algorithm_id, start_date)
                
                if not performance or performance['total_predictions'] < 10:
                    logger.warning(f"Yeterli performans verisi yok: {algorithm_name}")
                    continue
                
                # Algoritma doğruluğunu getir
                accuracy = performance['accuracy']
                
                # Son birkaç gündeki eğilimi hesapla
                recent_days = min(3, days)
                recent_start_date = (datetime.now() - timedelta(days=recent_days)).isoformat()
                recent_performance = self.db_manager.get_algorithm_performance_since_date(algorithm_id, recent_start_date)
                
                recent_accuracy = 0.0
                trend = 0.0
                
                if recent_performance and recent_performance['total_predictions'] > 0:
                    recent_accuracy = recent_performance['accuracy']
                    trend = recent_accuracy - accuracy if accuracy > 0 else 0
                
                # Performans verilerini topla
                performance_data.append({
                    'algorithm_id': algorithm_id,
                    'algorithm': algorithm,
                    'name': algorithm_name,
                    'current_weight': current_weight,
                    'accuracy': accuracy,
                    'recent_accuracy': recent_accuracy,
                    'trend': trend
                })
            
            # Performans verilerini doğruluk sırasına göre sırala
            performance_data.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Toplam ağırlık havuzu
            total_weight_pool = len(performance_data) * self.default_weight
            
            # Algoritmaların göreceli performansına göre ağırlıkları dağıt
            total_accuracy = sum(data['accuracy'] for data in performance_data) if performance_data else 1.0
            
            for i, data in enumerate(performance_data):
                # Temel ağırlık (doğruluk oranına göre)
                base_weight = (data['accuracy'] / total_accuracy) * total_weight_pool if total_accuracy > 0 else self.default_weight
                
                # Eğilim faktörü
                trend_factor = 1.0 + data['trend'] * 2.0  # Pozitif eğilim ağırlığı artırır
                
                # Yeni ağırlık
                new_weight = base_weight * trend_factor
                
                # Sınırları kontrol et
                new_weight = max(min(new_weight, self.max_weight), self.min_weight)
                
                # Ağırlığı güncelle
                data['algorithm'].set_weight(new_weight)
                
                # Veritabanını güncelle
                self.db_manager.update_algorithm_weight(data['algorithm_id'], new_weight)
                
                # Sonuçları kaydet
                results[data['name']] = {
                    'old_weight': data['current_weight'],
                    'new_weight': new_weight,
                    'accuracy': data['accuracy'],
                    'recent_accuracy': data['recent_accuracy'],
                    'trend': data['trend'],
                    'rank': i + 1
                }
                
                logger.info(f"Algoritma ağırlığı güncellendi: {data['name']}, {data['current_weight']:.2f} -> {new_weight:.2f}")
            
            return results
        except Exception as e:
            logger.error(f"Uyarlanabilir optimizasyon hatası: {str(e)}")
            return {}
    
    def _balanced_optimization(self, days=7):
        """
        Dengeli ağırlık optimizasyonu
        
        Performansa göre ağırlıkları belirlerken, algoritma çeşitliliğini de korur.
        
        Args:
            days (int): Performans analizinde kullanılacak gün sayısı
        
        Returns:
            dict: Optimizasyon sonuçları
        """
        try:
            results = {}
            
            # Tüm algoritmaları getir
            algorithms = self.prediction_engine.algorithms
            algorithm_data = []
            
            # Her algoritmanın son performansını getir
            for algorithm in algorithms:
                algorithm_name = algorithm.name
                current_weight = algorithm.weight
                
                # Algoritma tipini belirle
                algorithm_type = None
                if "Pattern" in algorithm_name or "Sequence" in algorithm_name:
                    algorithm_type = "pattern"
                elif "Statistical" in algorithm_name or "Bayes" in algorithm_name:
                    algorithm_type = "statistical"
                elif "Markov" in algorithm_name:
                    algorithm_type = "markov"
                elif "Monte" in algorithm_name:
                    algorithm_type = "monte_carlo"
                elif "Clustering" in algorithm_name:
                    algorithm_type = "clustering"
                else:
                    algorithm_type = "other"
                
                # Algoritma ID'sini bul
                algorithm_info = self.db_manager.get_algorithm_by_name(algorithm_name)
                
                if not algorithm_info:
                    logger.warning(f"Algoritma veritabanında bulunamadı: {algorithm_name}")
                    continue
                
                algorithm_id = algorithm_info['id']
                
                # Son N günlük performansı getir
                start_date = (datetime.now() - timedelta(days=days)).isoformat()
                performance = self.db_manager.get_algorithm_performance_since_date(algorithm_id, start_date)
                
                if not performance or performance['total_predictions'] < 10:
                    logger.warning(f"Yeterli performans verisi yok: {algorithm_name}")
                    accuracy = 0.5  # Varsayılan doğruluk
                else:
                    accuracy = performance['accuracy']
                
                # Algoritma verilerini topla
                algorithm_data.append({
                    'algorithm_id': algorithm_id,
                    'algorithm': algorithm,
                    'name': algorithm_name,
                    'type': algorithm_type,
                    'current_weight': current_weight,
                    'accuracy': accuracy
                })
            
            # Algoritma gruplarını oluştur
            algorithm_groups = {}
            for data in algorithm_data:
                group = data['type']
                if group not in algorithm_groups:
                    algorithm_groups[group] = []
                algorithm_groups[group].append(data)
            
            # Her grupta en iyi algoritmayı belirle ve ağırlıkları ayarla
            for group, group_algorithms in algorithm_groups.items():
                # Grubu doğruluk sırasına göre sırala
                group_algorithms.sort(key=lambda x: x['accuracy'], reverse=True)
                
                # Grup içindeki en iyi algoritmalara daha yüksek ağırlık ver
                for i, data in enumerate(group_algorithms):
                    # En iyi algoritma için en yüksek ağırlık
                    if i == 0:
                        new_weight = min(data['current_weight'] * 1.2, self.max_weight)
                    # İkinci en iyi için mevcut ağırlık
                    elif i == 1:
                        new_weight = data['current_weight']
                    # Diğerleri için azalan ağırlık
                    else:
                        new_weight = max(data['current_weight'] * 0.8, self.min_weight)
                    
                    # Ağırlığı güncelle
                    data['algorithm'].set_weight(new_weight)
                    
                    # Veritabanını güncelle
                    self.db_manager.update_algorithm_weight(data['algorithm_id'], new_weight)
                    
                    # Sonuçları kaydet
                    results[data['name']] = {
                        'old_weight': data['current_weight'],
                        'new_weight': new_weight,
                        'accuracy': data['accuracy'],
                        'type': data['type'],
                        'group_rank': i + 1
                    }
                    
                    logger.info(f"Algoritma ağırlığı güncellendi: {data['name']}, {data['current_weight']:.2f} -> {new_weight:.2f}")
            
            return results
        except Exception as e:
            logger.error(f"Dengeli optimizasyon hatası: {str(e)}")
            return {}
    
    def _random_search_optimization(self, days=7):
        """
        Rastgele arama ağırlık optimizasyonu
        
        Rastgele ağırlık kombinasyonları deneyerek en iyi sonucu bulmaya çalışır.
        
        Args:
            days (int): Performans analizinde kullanılacak gün sayısı
        
        Returns:
            dict: Optimizasyon sonuçları
        """
        try:
            results = {}
            
            # Tüm algoritmaları getir
            algorithms = self.prediction_engine.algorithms
            current_weights = {algorithm.name: algorithm.weight for algorithm in algorithms}
            
            # Mevcut ağırlıklar ile performansı ölç
            current_performance = self._evaluate_weights(days=days)
            best_performance = current_performance
            best_weights = current_weights.copy()
            
            # Rastgele arama iterasyonları
            max_iterations = 20
            
            for iteration in range(max_iterations):
                # Rastgele ağırlık ayarlamaları
                test_weights = {}
                
                for algorithm in algorithms:
                    # Rastgele bir değişiklik (-30% ile +30% arası)
                    change_factor = 1.0 + random.uniform(-0.3, 0.3)
                    new_weight = algorithm.weight * change_factor
                    
                    # Sınırları kontrol et
                    new_weight = max(min(new_weight, self.max_weight), self.min_weight)
                    test_weights[algorithm.name] = new_weight
                
                # Ağırlıkları geçici olarak ayarla
                self._apply_weights(test_weights)
                
                # Performansı ölç
                test_performance = self._evaluate_weights(days=days)
                
                # Daha iyi bir sonuç bulunduysa güncelle
                if test_performance > best_performance:
                    best_performance = test_performance
                    best_weights = test_weights.copy()
                    logger.info(f"İterasyon {iteration+1}: Daha iyi ağırlık kombinasyonu bulundu (Performans: {best_performance:.4f})")
                
                # Orijinal ağırlıkları geri yükle
                self._apply_weights(current_weights)
            
            # En iyi ağırlıkları uygula
            for algorithm in algorithms:
                algorithm_name = algorithm.name
                current_weight = current_weights[algorithm_name]
                new_weight = best_weights[algorithm_name]
                
                # Ağırlığı kalıcı olarak güncelle
                algorithm.set_weight(new_weight)
                
                # Algoritma ID'sini bul
                algorithm_info = self.db_manager.get_algorithm_by_name(algorithm_name)
                
                if algorithm_info:
                    algorithm_id = algorithm_info['id']
                    # Veritabanını güncelle
                    self.db_manager.update_algorithm_weight(algorithm_id, new_weight)
                
                # Sonuçları kaydet
                results[algorithm_name] = {
                    'old_weight': current_weight,
                    'new_weight': new_weight,
                    'change_percentage': ((new_weight / current_weight) - 1.0) * 100 if current_weight > 0 else 0
                }
                
                logger.info(f"Algoritma ağırlığı güncellendi: {algorithm_name}, {current_weight:.2f} -> {new_weight:.2f}")
            
            return results
        except Exception as e:
            logger.error(f"Rastgele arama optimizasyonu hatası: {str(e)}")
            return {}
    
    def _evaluate_weights(self, days=7):
        """
        Mevcut ağırlıkların performansını değerlendirir
        
        Args:
            days (int): Performans analizinde kullanılacak gün sayısı
        
        Returns:
            float: Genel performans skoru
        """
        try:
            # Son N günlük sonuçları getir
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            recent_results = self.db_manager.get_results_since_date(start_date)
            
            if not recent_results or len(recent_results) < 10:
                logger.warning(f"Yeterli sonuç verisi yok: {len(recent_results) if recent_results else 0}/10")
                return 0.0
            
            # Son sonuçlar için tahminleri değerlendir
            correct_count = 0
            weighted_correct = 0.0
            total_weight = 0.0
            
            for result in recent_results:
                # Son 20 sonucu al
                previous_results = self.db_manager.get_results_before_date(result['timestamp'], limit=20)
                previous_results_str = [r['result'] for r in previous_results]
                
                # Tahmin yap
                prediction = self.prediction_engine.predict({'last_results': previous_results_str}, save_prediction=False)
                
                if prediction and prediction['prediction'] == result['result']:
                    correct_count += 1
                    weighted_correct += prediction['confidence']
                
                total_weight += 1.0
            
            # Performans skorunu hesapla
            accuracy = correct_count / len(recent_results) if recent_results else 0.0
            weighted_accuracy = weighted_correct / total_weight if total_weight > 0 else 0.0
            
            # Ağırlıklı ve normal doğruluğun karışımı
            performance_score = (accuracy * 0.7) + (weighted_accuracy * 0.3)
            
            return performance_score
        except Exception as e:
            logger.error(f"Ağırlık değerlendirme hatası: {str(e)}")
            return 0.0
    
    def _apply_weights(self, weights):
        """
        Algoritmalara belirli ağırlıkları uygular (geçici)
        
        Args:
            weights (dict): Algoritma adı -> ağırlık eşleşmesi
        """
        for algorithm in self.prediction_engine.algorithms:
            if algorithm.name in weights:
                algorithm.weight = weights[algorithm.name]
    
    def reset_weights(self):
        """
        Tüm algoritma ağırlıklarını varsayılan değerlere sıfırlar
        
        Returns:
            dict: Sıfırlama sonuçları
        """
        try:
            results = {}
            
            for algorithm in self.prediction_engine.algorithms:
                algorithm_name = algorithm.name
                current_weight = algorithm.weight
                
                # Varsayılan ağırlığa sıfırla
                algorithm.set_weight(self.default_weight)
                
                # Algoritma ID'sini bul
                algorithm_info = self.db_manager.get_algorithm_by_name(algorithm_name)
                
                if algorithm_info:
                    algorithm_id = algorithm_info['id']
                    # Veritabanını güncelle
                    self.db_manager.update_algorithm_weight(algorithm_id, self.default_weight)
                
                # Sonuçları kaydet
                results[algorithm_name] = {
                    'old_weight': current_weight,
                    'new_weight': self.default_weight,
                    'reset': True
                }
                
                logger.info(f"Algoritma ağırlığı sıfırlandı: {algorithm_name}, {current_weight:.2f} -> {self.default_weight:.2f}")
            
            return results
        except Exception as e:
            logger.error(f"Ağırlık sıfırlama hatası: {str(e)}")
            return {}