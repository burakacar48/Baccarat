#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dönemsel (Cyclical) Analiz
------------------------
Baccarat sonuçlarında belirli aralıklarla tekrar eden döngüleri arar.
"""

import logging
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class CyclicalAnalysis(BaseAlgorithm):
    """
    Dönemsel (Cyclical) Analiz
    
    Baccarat sonuçlarında belirli aralıklarla tekrar eden döngüleri arar.
    Farklı uzunluktaki döngüleri test ederek en güçlü döngüsel paterni bulmaya çalışır.
    """
    
    def __init__(self, weight=1.0, db_manager=None, cycle_lengths=[5, 7, 9, 11]):
        """
        CyclicalAnalysis sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            cycle_lengths (list): Test edilecek döngü uzunlukları listesi
        """
        super().__init__(name="Cyclical Analysis", weight=weight)
        self.db_manager = db_manager
        self.cycle_lengths = cycle_lengths
        
        logger.info(f"Dönemsel Analiz başlatıldı: Döngü uzunlukları={cycle_lengths}")
    
    def predict(self, data):
        """
        Verilen verilere göre tahmin yapar
        
        Args:
            data (dict): Tahmin için gerekli veriler
                - last_results (list): Son oyun sonuçları listesi
                - custom_cycles (list, optional): Özel döngü uzunlukları
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - confidence (float): Güven skoru (0-1 arası)
                - details (dict): Ek detaylar
        """
        # Veri doğrulama
        if 'last_results' not in data or not data['last_results']:
            logger.warning("CyclicalAnalysis: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Özel döngü uzunlukları varsa kullan
        cycle_lengths = data.get('custom_cycles', self.cycle_lengths)
        
        # Tüm sonuçları al
        all_results = self.db_manager.get_all_results()
        
        if len(all_results) < max(cycle_lengths) * 2:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"CyclicalAnalysis: Yetersiz veri - {len(all_results)}/{max(cycle_lengths) * 2}")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # En güçlü döngüyü bul
        best_cycle = None
        best_score = 0
        best_prediction = None
        cycle_details = {}
        
        for cycle_length in cycle_lengths:
            # Son 'cycle_length' kadar sonuç
            cycle_pattern = all_results[-cycle_length:]
            
            # Bu döngü kalıbını geçmişte ara
            matches = 0
            total = 0
            next_outcomes = []
            
            for i in range(len(all_results) - cycle_length - 1):
                pattern_match = True
                for j in range(cycle_length):
                    if all_results[i+j] != cycle_pattern[j]:
                        pattern_match = False
                        break
                
                if pattern_match:
                    matches += 1
                    next_outcomes.append(all_results[i+cycle_length])
                
                total += 1
            
            # Döngü skoru
            if total > 0:
                cycle_score = matches / total
                
                if matches > 0 and cycle_score > best_score:
                    # En sık sonucu bul
                    counter = Counter(next_outcomes)
                    most_common = counter.most_common(1)[0]
                    prediction = most_common[0]
                    confidence = most_common[1] / len(next_outcomes)
                    
                    best_cycle = cycle_length
                    best_score = cycle_score
                    best_prediction = {
                        'prediction': prediction,
                        'confidence': confidence
                    }
                
                # Döngü detaylarını kaydet
                cycle_details[cycle_length] = {
                    'matches': matches,
                    'total': total,
                    'score': cycle_score,
                    'next_outcomes': dict(Counter(next_outcomes))
                }
        
        # En iyi döngü bulunduysa kullan
        if best_prediction:
            # Detaylar
            details = {
                'best_cycle_length': best_cycle,
                'best_cycle_score': best_score,
                'all_cycles': cycle_details
            }
            
            logger.debug(f"CyclicalAnalysis tahmini: {best_prediction['prediction']}, Güven: {best_prediction['confidence']:.4f}, Döngü: {best_cycle}")
            return {
                'prediction': best_prediction['prediction'],
                'confidence': best_prediction['confidence'],
                'details': details
            }
        
        # Anlamlı döngü bulunamadıysa varsayılan tahmin
        logger.warning("CyclicalAnalysis: Anlamlı döngü bulunamadı")
        return {
            'prediction': 'B',
            'confidence': 0.5
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
    
    def set_cycle_lengths(self, cycle_lengths):
        """
        Test edilecek döngü uzunluklarını ayarlar
        
        Args:
            cycle_lengths (list): Yeni döngü uzunlukları listesi
        """
        if not cycle_lengths or min(cycle_lengths) < 2:
            logger.warning(f"CyclicalAnalysis: Geçersiz döngü uzunlukları: {cycle_lengths}, minimum 2 olmalı")
            return
        
        self.cycle_lengths = cycle_lengths
        logger.info(f"CyclicalAnalysis döngü uzunlukları {cycle_lengths} olarak ayarlandı")