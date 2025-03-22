#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Korelasyon Tabanlı Model
---------------------
Player, Banker ve Tie sonuçları arasındaki korelasyonları araştırır.
"""

import logging
import numpy as np
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class CorrelationModel(BaseAlgorithm):
    """
    Korelasyon Tabanlı Model
    
    Player, Banker ve Tie sonuçları arasındaki korelasyonları araştırır ve
    desen kalıplarına dayalı olarak gelecek tahminleri yapar.
    """
    
    def __init__(self, weight=1.0, db_manager=None, window_size=50):
        """
        CorrelationModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            window_size (int): Korelasyon analizinin yapılacağı pencere boyutu
        """
        super().__init__(name="Correlation Model", weight=weight)
        self.db_manager = db_manager
        self.window_size = window_size
        
        logger.info(f"Korelasyon Modeli başlatıldı: Pencere boyutu={window_size}")
    
    def predict(self, data):
        """
        Verilen verilere göre tahmin yapar
        
        Args:
            data (dict): Tahmin için gerekli veriler
                - last_results (list): Son oyun sonuçları listesi
                - window_size (int, optional): Özel pencere boyutu
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - confidence (float): Güven skoru (0-1 arası)
                - details (dict): Ek detaylar
        """
        # Veri doğrulama
        if 'last_results' not in data or not data['last_results']:
            logger.warning("CorrelationModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Son N sonucu al
        last_results = data.get('last_results', [])
        
        if len(last_results) < 20:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"CorrelationModel: Yetersiz veri - {len(last_results)}/20")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Pencere boyutunu kullan veya özel değeri al
        window_size = data.get('window_size', self.window_size)
        
        # Son 'window_size' kadar sonucu analiz et
        window = last_results[-window_size:] if len(last_results) >= window_size else last_results
        
        # Korelasyon matrisini oluştur
        correlation_matrix = self._build_correlation_matrix(window)
        
        # Son 3 sonucu kontrol et
        patterns = {}
        
        for length in [1, 2, 3]:
            if len(last_results) >= length:
                pattern = ''.join(last_results[-length:])
                patterns[length] = pattern
        
        # Korelasyonlara göre tahmin yap
        probabilities = {'P': 0.0, 'B': 0.0, 'T': 0.0}
        
        # Her kalıp için korelasyonları kontrol et
        for length, pattern in patterns.items():
            if pattern in correlation_matrix:
                for outcome, corr in correlation_matrix[pattern].items():
                    # Korelasyonu olasılığa dönüştür (0-1 arası)
                    prob = (corr + 1) / 2  # Korelasyon -1 ile 1 arasında olduğundan
                    probabilities[outcome] += prob * (length / 6)  # Uzun kalıplara daha fazla ağırlık ver
        
        # Normalize et
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        # En olası sonucu bul
        prediction = max(probabilities, key=probabilities.get)
        confidence = probabilities[prediction]
        
        # Detaylar
        details = {
            'patterns': patterns,
            'window_size': len(window),
            'probabilities': probabilities
        }
        
        logger.debug(f"CorrelationModel tahmini: {prediction}, Güven: {confidence:.4f}")
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': details
        }
    
    def _build_correlation_matrix(self, results):
        """
        Korelasyon matrisini oluşturur
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            dict: Korelasyon matrisi
        """
        matrix = {}
        
        # Tüm olası 1, 2 ve 3 uzunluğundaki kalıplar için
        for length in [1, 2, 3]:
            for i in range(len(results) - length):
                pattern = ''.join(results[i:i+length])
                
                if pattern not in matrix:
                    matrix[pattern] = {'P': 0, 'B': 0, 'T': 0}
                
                if i + length < len(results):
                    next_outcome = results[i+length]
                    matrix[pattern][next_outcome] += 1
        
        # Sayıları korelasyonlara dönüştür
        for pattern, counts in matrix.items():
            total = sum(counts.values())
            if total > 0:
                baseline = {'P': 0.4932, 'B': 0.5068, 'T': 0.0955}  # Teorik olasılıklar
                for outcome in counts:
                    observed = counts[outcome] / total
                    expected = baseline[outcome]
                    # Korelasyon: -1 ile 1 arası (-1: negatif, 0: nötr, 1: pozitif)
                    matrix[pattern][outcome] = 2 * (observed - expected)
        
        return matrix
    
    def get_confidence(self, data):
        """
        Tahminin güven skorunu hesaplar
        
        Args:
            data (dict): Tahmin için gerekli veriler
        
        Returns:
            float: Güven skoru (0-1 arası)
        """
        prediction_result = self.predict(data)
        return prediction_result['confidence']
    
    def set_window_size(self, window_size):
        """
        Korelasyon analizi pencere boyutunu ayarlar
        
        Args:
            window_size (int): Yeni pencere boyutu
        """
        if window_size < 10:
            logger.warning(f"CorrelationModel: Geçersiz pencere boyutu: {window_size}, minimum 10 olmalı")
            return
        
        self.window_size = window_size
        logger.info(f"CorrelationModel pencere boyutu {window_size} olarak ayarlandı")