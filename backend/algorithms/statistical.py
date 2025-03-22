#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
İstatistiksel Model
-----------------
Baccarat'ın teorik olasılıklarını ve geçmiş verilerden elde edilen dağılımları kullanarak tahmin yapar.
"""

import logging
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class StatisticalModel(BaseAlgorithm):
    """
    İstatistiksel Model
    
    Baccarat'ın teorik olasılıklarını ve geçmiş verilerden elde edilen 
    dağılımları kullanarak tahmin yapar. Ortalamaya dönme eğilimini analiz eder.
    """
    
    def __init__(self, weight=1.0, db_manager=None, window_size=50):
        """
        StatisticalModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            window_size (int): İstatistiksel analizin yapılacağı pencere boyutu
        """
        super().__init__(name="Statistical Model", weight=weight)
        self.db_manager = db_manager
        self.window_size = window_size
        
        # Teorik olasılıklar
        self.theoretical_probs = {'P': 0.4932, 'B': 0.5068, 'T': 0.0955}
        
        logger.info(f"İstatistiksel Model başlatıldı: Pencere boyutu={window_size}")
    
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
            logger.warning("StatisticalModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',  # En yüksek teorik olasılık
                'confidence': self.theoretical_probs['B'],
                'details': {'theoretical': self.theoretical_probs}
            }
        
        # Son N sonucu al
        last_results = data.get('last_results', [])
        window_size = data.get('window_size', self.window_size)
        
        # Pencere boyutunu son sonuç sayısıyla sınırla
        window_size = min(window_size, len(last_results))
        
        if window_size < 10:
            # Yeterli veri yoksa teorik olasılıkları kullan
            logger.warning(f"StatisticalModel: Yetersiz veri - {window_size}/10")
            prediction = 'B'  # En yüksek teorik olasılık
            return {
                'prediction': prediction,
                'confidence': self.theoretical_probs[prediction],
                'details': {'theoretical': self.theoretical_probs}
            }
        
        # Son pencere boyutu kadar sonucu al
        window_results = last_results[-window_size:]
        
        # Gerçek dağılımı hesapla
        counter = Counter(window_results)
        total = len(window_results)
        actual_probs = {k: v/total for k, v in counter.items()}
        
        # Eksik sonuçlara 0 olasılık ata
        for outcome in ['P', 'B', 'T']:
            actual_probs[outcome] = actual_probs.get(outcome, 0)
        
        # Ortalamaya dönme eğilimini hesapla
        probabilities = {}
        for outcome in ['P', 'B', 'T']:
            theoretical = self.theoretical_probs[outcome]
            actual = actual_probs[outcome]
            
            # Sapma yönüne göre tahmin üret
            if actual > theoretical:
                # Normalden fazla görülmüş, azalma beklenir
                probabilities[outcome] = theoretical - (actual - theoretical) * 0.5
            else:
                # Normalden az görülmüş, artış beklenir
                probabilities[outcome] = theoretical + (theoretical - actual) * 0.5
        
        # En olası sonucu belirle
        prediction = max(probabilities, key=probabilities.get)
        confidence = probabilities[prediction]
        
        # Detaylar
        details = {
            'theoretical': self.theoretical_probs,
            'actual': actual_probs,
            'adjusted': probabilities
        }
        
        logger.debug(f"StatisticalModel tahmini: {prediction}, Güven: {confidence:.4f}")
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': details
        }
    
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
        İstatistiksel analiz pencere boyutunu ayarlar
        
        Args:
            window_size (int): Yeni pencere boyutu
        """
        if window_size < 5:
            logger.warning(f"StatisticalModel: Geçersiz pencere boyutu: {window_size}, minimum 5 olmalı")
            return
        
        self.window_size = window_size
        logger.info(f"StatisticalModel pencere boyutu {window_size} olarak ayarlandı")