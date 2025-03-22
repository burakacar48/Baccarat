#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sıralı Desen Analizi
-----------------
Baccarat'ta bazen uzun seriler (streak) oluşabilir. Bu algoritma, hem devam eden serileri 
hem de seri kırılmalarını analiz eder.
"""

import logging
import random
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class SequenceAnalysis(BaseAlgorithm):
    """
    Sıralı Desen Analizi
    
    Baccarat'ta bazen uzun seriler (streak) oluşabilir. Bu algoritma, hem devam eden serileri 
    hem de seri kırılmalarını analiz eder.
    """
    
    def __init__(self, weight=1.0, db_manager=None, sequence_length=5):
        """
        SequenceAnalysis sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            sequence_length (int): Maksimum kontrol edilecek seri uzunluğu
        """
        super().__init__(name="Sequence Analysis", weight=weight)
        self.db_manager = db_manager
        self.sequence_length = sequence_length
        
        logger.info(f"Sıralı Desen Analizi başlatıldı: Max seri uzunluğu={sequence_length}")
    
    def predict(self, data):
        """
        Verilen verilere göre tahmin yapar
        
        Args:
            data (dict): Tahmin için gerekli veriler
                - last_results (list): Son oyun sonuçları listesi
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - confidence (float): Güven skoru (0-1 arası)
                - details (dict): Ek detaylar
        """
        # Veri doğrulama
        if 'last_results' not in data or not data['last_results']:
            logger.warning("SequenceAnalysis: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B', 
                'confidence': 0.5
            }
        
        # Son sonuçları al
        last_results = data.get('last_results', [])
        
        if len(last_results) < 3:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"SequenceAnalysis: Yetersiz veri - {len(last_results)}/3")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Mevcut seriyi kontrol et
        current_streak = 1
        streak_value = last_results[-1]
        
        for i in range(len(last_results)-2, -1, -1):
            if last_results[i] == streak_value:
                current_streak += 1
            else:
                break
        
        # Geçmiş verilerden benzer seri uzunluklarını bul
        all_results = self.db_manager.get_all_results()
        streak_continuations = []
        streak_breaks = []
        
        for i in range(len(all_results) - current_streak - 1):
            # Seri bul
            streak_found = True
            for j in range(current_streak):
                if i+j >= len(all_results) or all_results[i+j] != streak_value:
                    streak_found = False
                    break
            
            if streak_found and i+current_streak < len(all_results):
                next_result = all_results[i+current_streak]
                if next_result == streak_value:
                    streak_continuations.append(next_result)
                else:
                    streak_breaks.append(next_result)
        
        # Seri devam mı edecek yoksa kırılacak mı?
        total_samples = len(streak_continuations) + len(streak_breaks)
        
        # Detaylar
        details = {
            'current_streak': current_streak,
            'streak_value': streak_value,
            'continuations': len(streak_continuations),
            'breaks': len(streak_breaks),
            'total_samples': total_samples
        }
        
        if total_samples < 5:
            # Yeterli örnek yoksa varsayılan tahmin
            logger.warning(f"SequenceAnalysis: Yetersiz örnek sayısı - {total_samples}/5")
            return {
                'prediction': 'B',
                'confidence': 0.5,
                'details': details
            }
        
        if len(streak_continuations) > len(streak_breaks):
            # Seri devam ediyor
            prediction = streak_value
            confidence = len(streak_continuations) / total_samples
            
            logger.debug(f"SequenceAnalysis: Seri devam ediyor - {streak_value} x {current_streak}, Tahmin: {prediction}, Güven: {confidence:.4f}")
        else:
            # Seri kırılıyor
            if len(streak_breaks) > 0:
                counter = Counter(streak_breaks)
                prediction = counter.most_common(1)[0][0]
                confidence = counter.most_common(1)[0][1] / len(streak_breaks)
                
                logger.debug(f"SequenceAnalysis: Seri kırılıyor - {streak_value} x {current_streak}, Tahmin: {prediction}, Güven: {confidence:.4f}")
            else:
                # Alternatif sonuçlardan birini seç
                outcomes = ['P', 'B', 'T']
                outcomes.remove(streak_value)
                prediction = random.choice(outcomes)
                confidence = 0.5
                
                logger.debug(f"SequenceAnalysis: Seri kırılma örnekleri yok - Tahmin: {prediction}, Güven: {confidence:.4f}")
        
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
    
    def set_sequence_length(self, length):
        """
        Maksimum kontrol edilecek seri uzunluğunu ayarlar
        
        Args:
            length (int): Yeni uzunluk değeri
        """
        if length < 2:
            logger.warning(f"SequenceAnalysis: Geçersiz uzunluk değeri: {length}, minimum 2 olmalı")
            return
        
        self.sequence_length = length
        logger.info(f"SequenceAnalysis seri uzunluğu {length} olarak ayarlandı")