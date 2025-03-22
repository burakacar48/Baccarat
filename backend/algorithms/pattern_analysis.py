#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Desen Analizi Algoritması
-------------------------
Belirli bir uzunluktaki son oyun sonuçlarını (örneğin son 3-5 oyun) bir desen olarak değerlendirir
ve bu desenin geçmişte hangi sonuçlarla devam ettiğini analiz eder.
"""

import logging
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class PatternAnalysis(BaseAlgorithm):
    """
    Desen Analizi Algoritması
    
    Belirli bir uzunluktaki son oyun sonuçlarını (desen) alır ve tarihsel verilerde
    bu desenin hangi sonuçlarla devam ettiğini analiz ederek tahmin üretir.
    """
    
    def __init__(self, weight=1.0, db_manager=None, min_samples=5, pattern_length=3):
        """
        PatternAnalysis sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            min_samples (int): Analiz için minimum örnek sayısı
            pattern_length (int): Desen uzunluğu (son kaç oyun sonucunun analiz edileceği)
        """
        super().__init__(name="Pattern Analysis", weight=weight)
        self.db_manager = db_manager
        self.min_samples = min_samples
        self.pattern_length = pattern_length
    
    def predict(self, data):
        """
        Verilen verilere göre tahmin yapar
        
        Args:
            data (dict): Tahmin için gerekli veriler
                - last_results (list): Son oyun sonuçları listesi
                - custom_pattern_length (int, optional): Özel desen uzunluğu
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - confidence (float): Güven skoru (0-1 arası)
                - pattern (str): Analiz edilen desen
                - samples (int): Bulunan örnek sayısı
                - top_candidates (list): En olası sonuçlar ve olasılıkları
        """
        # Veri doğrulama
        if 'last_results' not in data or not data['last_results']:
            logger.warning("PatternAnalysis: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',  # Varsayılan olarak Banker'ı döndür (biraz daha yüksek teorik olasılık)
                'confidence': 0.1,
                'pattern': '',
                'samples': 0,
                'top_candidates': [('B', 0.5068), ('P', 0.4932)]
            }
        
        # Özel desen uzunluğu varsa kullan
        pattern_length = data.get('custom_pattern_length', self.pattern_length)
        
        # Son sonuçları al
        last_results = data['last_results']
        
        # Desen uzunluğu için son sonuçları kontrol et
        if len(last_results) < pattern_length:
            logger.warning(f"PatternAnalysis: Yetersiz sonuç sayısı - {len(last_results)}/{pattern_length}")
            # Yeterli sonuç yoksa basit bir tahmin yap
            return {
                'prediction': 'B',
                'confidence': 0.2,
                'pattern': ''.join(last_results),
                'samples': 0,
                'top_candidates': [('B', 0.5068), ('P', 0.4932)]
            }
        
        # Deseni oluştur (son N sonuç)
        current_pattern = last_results[-pattern_length:]
        pattern_str = ''.join(current_pattern)
        
        logger.debug(f"PatternAnalysis: Aranan desen: {pattern_str}")
        
        # Veritabanından tüm sonuçları getir
        all_results = self.db_manager.get_all_results()
        
        # Benzer desenleri bul ve devamındaki sonuçları analiz et
        next_results = []
        
        for i in range(len(all_results) - pattern_length):
            # Olası bir desen bul
            possible_pattern = all_results[i:i+pattern_length]
            possible_pattern_str = ''.join(possible_pattern)
            
            # Desen eşleşiyorsa ve devamı varsa
            if possible_pattern_str == pattern_str and i + pattern_length < len(all_results):
                next_result = all_results[i + pattern_length]
                next_results.append(next_result)
        
        # Yeterli örnek bulunamadıysa
        if len(next_results) < self.min_samples:
            logger.warning(f"PatternAnalysis: Yetersiz örnek sayısı - {len(next_results)}/{self.min_samples}")
            return {
                'prediction': 'B',
                'confidence': 0.3,
                'pattern': pattern_str,
                'samples': len(next_results),
                'top_candidates': [('B', 0.5068), ('P', 0.4932)]
            }
        
        # Sonuçları say ve en yaygın olanı bul
        result_counter = Counter(next_results)
        total_samples = len(next_results)
        
        # En yaygın sonuçları olasılıklarıyla birlikte al
        top_candidates = [(result, count / total_samples) for result, count in result_counter.most_common()]
        
        # En olası sonucu ve güven skorunu belirle
        prediction = top_candidates[0][0]
        confidence = top_candidates[0][1]
        
        # Sonucu logla
        logger.info(f"PatternAnalysis: Desen={pattern_str}, Tahmin={prediction}, Güven={confidence:.2f}, Örnek={total_samples}")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'pattern': pattern_str,
            'samples': total_samples,
            'top_candidates': top_candidates
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
    
    def set_pattern_length(self, length):
        """
        Desen uzunluğunu ayarlar
        
        Args:
            length (int): Yeni desen uzunluğu
        """
        if length < 1:
            logger.warning(f"PatternAnalysis: Geçersiz desen uzunluğu: {length}, minimum 1 olmalı")
            return
        
        self.pattern_length = length
        logger.info(f"PatternAnalysis: Desen uzunluğu {length} olarak ayarlandı")
    
    def set_min_samples(self, min_samples):
        """
        Minimum örnek sayısını ayarlar
        
        Args:
            min_samples (int): Yeni minimum örnek sayısı
        """
        if min_samples < 1:
            logger.warning(f"PatternAnalysis: Geçersiz minimum örnek sayısı: {min_samples}, minimum 1 olmalı")
            return
        
        self.min_samples = min_samples
        logger.info(f"PatternAnalysis: Minimum örnek sayısı {min_samples} olarak ayarlandı")