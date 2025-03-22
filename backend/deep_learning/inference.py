#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Çıkarım (Inference) Modülü
------------------------
LSTM modeli ile tahmin yapmayı sağlar.
"""

import logging
import numpy as np
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class Inference:
    """
    Derin öğrenme modeliyle tahmin (inference) sınıfı
    
    Eğitilmiş LSTM modelini kullanarak yeni tahminler yapar.
    """
    
    def __init__(self, model, data_preparation):
        """
        Inference sınıfını başlatır
        
        Args:
            model: LSTM model nesnesi
            data_preparation: DataPreparation nesnesi
        """
        self.model = model
        self.data_preparation = data_preparation
        self.last_prediction_time = None
        self.prediction_cache = None
        self.cache_timeout = 60  # Saniye cinsinden önbellek süresi
        
        logger.info("Çıkarım (Inference) modülü başlatıldı")
    
    def predict(self, data=None, use_cache=True):
        """
        Verilen veriler için tahmin yapar
        
        Args:
            data: Tahmin için giriş verileri (None ise son verileri otomatik hazırlar)
            use_cache (bool): Önbellek kullanılsın mı
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - probabilities (dict): Her sınıf için olasılıklar
                - confidence (float): Güven skoru (en yüksek olasılık)
        """
        try:
            # Önbelleği kontrol et
            current_time = time.time()
            if (use_cache and 
                self.last_prediction_time is not None and 
                self.prediction_cache is not None and 
                current_time - self.last_prediction_time < self.cache_timeout):
                
                logger.debug("Önbellekten tahmin sonucu kullanılıyor")
                return self.prediction_cache
            
            # Veriyi hazırla (eğer verilmemişse)
            if data is None:
                data = self.data_preparation.prepare_recent_data(
                    sequence_length=self.model.input_size
                )
                
                if data is None:
                    logger.error("Tahmin için yeterli veri yok")
                    return {
                        'prediction': 'B',
                        'probabilities': {'P': 0.4932, 'B': 0.5068, 'T': 0.0955},
                        'confidence': 0.5068
                    }
            
            # Modelin veri şeklini kontrol et ve düzelt
            if len(data.shape) == 2:  # (sequence_length, features)
                data = data.reshape(1, data.shape[0], data.shape[1])
            
            # Tahmin yap
            if self.model.model is None:
                logger.error("LSTM modeli yüklenemedi veya oluşturulmadı")
                return {
                    'prediction': 'B',
                    'probabilities': {'P': 0.4932, 'B': 0.5068, 'T': 0.0955},
                    'confidence': 0.5068
                }
            
            probabilities = self.model.model.predict(data, verbose=0)[0]
            
            # Sınıf indeksleri
            class_indices = {0: 'P', 1: 'B', 2: 'T'}
            
            # Olasılıklar sözlüğü
            prob_dict = {class_indices[i]: float(prob) for i, prob in enumerate(probabilities)}
            
            # En yüksek olasılıklı sınıfı belirle
            predicted_class_index = np.argmax(probabilities)
            predicted_class = class_indices[predicted_class_index]
            confidence = float(probabilities[predicted_class_index])
            
            # Sonucu oluştur
            result = {
                'prediction': predicted_class,
                'probabilities': prob_dict,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            # Önbelleğe al
            self.last_prediction_time = current_time
            self.prediction_cache = result
            
            logger.debug(f"LSTM tahmini: {predicted_class}, Güven: {confidence:.4f}")
            return result
        except Exception as e:
            logger.error(f"LSTM tahmin hatası: {str(e)}")
            return {
                'prediction': 'B',
                'probabilities': {'P': 0.4932, 'B': 0.5068, 'T': 0.0955},
                'confidence': 0.5068
            }
    
    def predict_batch(self, sequences):
        """
        Çoklu veri dizisi için tahmin yapar
        
        Args:
            sequences (ndarray): Tahmin edilecek veri dizileri
                                Shape: (batch_size, sequence_length, features)
        
        Returns:
            list: Tahmin sonuçları listesi
        """
        try:
            if self.model.model is None:
                logger.error("LSTM modeli yüklenemedi veya oluşturulmadı")
                return None
            
            # Toplu tahmin yap
            batch_probabilities = self.model.model.predict(sequences, verbose=0)
            
            # Sonuçları işle
            results = []
            class_indices = {0: 'P', 1: 'B', 2: 'T'}
            
            for probabilities in batch_probabilities:
                # Olasılıklar sözlüğü
                prob_dict = {class_indices[i]: float(prob) for i, prob in enumerate(probabilities)}
                
                # En yüksek olasılıklı sınıfı belirle
                predicted_class_index = np.argmax(probabilities)
                predicted_class = class_indices[predicted_class_index]
                confidence = float(probabilities[predicted_class_index])
                
                # Sonucu ekle
                results.append({
                    'prediction': predicted_class,
                    'probabilities': prob_dict,
                    'confidence': confidence
                })
            
            logger.debug(f"Toplu tahmin tamamlandı: {len(results)} tahmin")
            return results
        except Exception as e:
            logger.error(f"Toplu tahmin hatası: {str(e)}")
            return None
    
    def generate_feature_vector(self, sequence):
        """
        Verilen dizi için özellik vektörü oluşturur
        
        Args:
            sequence (list/ndarray): Sayısal sonuç dizisi
        
        Returns:
            dict: Özellik vektörü
        """
        try:
            result_map = {0: 'P', 1: 'B', 2: 'T'}
            sequence_str = [result_map.get(int(x), 'B') for x in sequence]
            
            # Basit istatistikler
            p_count = sequence.count(0) if hasattr(sequence, 'count') else np.sum(sequence == 0)
            b_count = sequence.count(1) if hasattr(sequence, 'count') else np.sum(sequence == 1)
            t_count = sequence.count(2) if hasattr(sequence, 'count') else np.sum(sequence == 2)
            
            total = len(sequence)
            p_ratio = p_count / total if total > 0 else 0
            b_ratio = b_count / total if total > 0 else 0
            t_ratio = t_count / total if total > 0 else 0
            
            # Seri analizi
            current_streak = 1
            max_streak = 1
            current_val = sequence[-1]
            
            for i in range(len(sequence)-2, -1, -1):
                if sequence[i] == current_val:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    break
            
            # Alternatif sonuç (P-B) analizi
            alternating_count = 0
            for i in range(1, len(sequence)):
                if (sequence[i] in [0, 1]) and (sequence[i-1] in [0, 1]) and (sequence[i] != sequence[i-1]):
                    alternating_count += 1
            
            alternating_ratio = alternating_count / (total - 1) if total > 1 else 0
            
            # Tahmin sonucu
            prediction = self.predict(np.array(sequence).reshape(1, len(sequence), 1))
            
            # Özellik vektörünü oluştur
            feature_vector = {
                'sequence': sequence_str,
                'sequence_numeric': sequence.tolist() if hasattr(sequence, 'tolist') else list(sequence),
                'stats': {
                    'player_count': int(p_count),
                    'banker_count': int(b_count),
                    'tie_count': int(t_count),
                    'player_ratio': float(p_ratio),
                    'banker_ratio': float(b_ratio),
                    'tie_ratio': float(t_ratio)
                },
                'patterns': {
                    'current_streak': int(current_streak),
                    'max_streak': int(max_streak),
                    'current_value': int(current_val),
                    'alternating_ratio': float(alternating_ratio)
                },
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities'],
                'timestamp': datetime.now().isoformat()
            }
            
            return feature_vector
        except Exception as e:
            logger.error(f"Özellik vektörü oluşturma hatası: {str(e)}")
            return None