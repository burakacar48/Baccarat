#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Markov Zinciri Modeli
-------------------
Baccarat sonuçlarını bir Markov süreci olarak modelleyerek, belirli bir durumdan
diğer durumlara geçiş olasılıklarını hesaplar.
"""

import logging
import random
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class MarkovModel(BaseAlgorithm):
    """
    Markov Zinciri Modeli
    
    Baccarat sonuçlarını bir Markov süreci olarak modelleyerek, belirli bir durumdan
    diğer durumlara geçiş olasılıklarını hesaplar.
    """
    
    def __init__(self, weight=1.0, db_manager=None, order=2):
        """
        MarkovModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            order (int): Markov zinciri mertebesi (kaç önceki durumun dikkate alınacağı)
        """
        super().__init__(name="Markov Chain", weight=weight)
        self.db_manager = db_manager
        self.order = order  # Markov zinciri mertebesi
        self.transitions = {}  # Geçiş matrisi
        
        logger.info(f"Markov Zinciri Modeli başlatıldı: Order={order}")
    
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
        """
        # Veri doğrulama
        if 'last_results' not in data or not data['last_results']:
            logger.warning("MarkovModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Son sonuçları al
        last_results = data['last_results']
        
        if len(last_results) < self.order:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"MarkovModel: Yetersiz veri - {len(last_results)}/{self.order}")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Geçiş matrisini oluştur/güncelle
        self._build_transition_matrix()
        
        # Mevcut durum (son 'order' kadar sonuç)
        current_state = ''.join(last_results[-self.order:])
        
        # Bu durumdan sonraki olası geçişleri bul
        if current_state in self.transitions:
            transitions = self.transitions[current_state]
            
            # En olası sonucu bul
            if transitions:
                prediction = max(transitions, key=transitions.get)
                confidence = transitions[prediction]
                
                logger.debug(f"MarkovModel tahmini: {prediction}, Güven: {confidence:.4f}, Durum: {current_state}")
                return {
                    'prediction': prediction,
                    'confidence': confidence
                }
        
        # Geçiş matrisi yoksa varsayılan tahmin
        logger.warning(f"MarkovModel: '{current_state}' durumu için geçiş bulunamadı")
        return {
            'prediction': 'B',
            'confidence': 0.5
        }
    
    def _build_transition_matrix(self):
        """
        Geçiş matrisini oluşturur
        """
        # Veritabanından tüm sonuçları getir
        all_results = self.db_manager.get_all_results()
        
        if len(all_results) <= self.order:
            logger.warning(f"MarkovModel: Geçiş matrisi için yetersiz veri - {len(all_results)}/{self.order+1}")
            return
        
        # Geçiş sayılarını hesapla
        transition_counts = {}
        
        for i in range(len(all_results) - self.order):
            state = ''.join(all_results[i:i+self.order])
            next_state = all_results[i+self.order]
            
            if state not in transition_counts:
                transition_counts[state] = {'P': 0, 'B': 0, 'T': 0}
            
            transition_counts[state][next_state] += 1
        
        # Olasılıklara dönüştür
        for state, counts in transition_counts.items():
            total = sum(counts.values())
            if total > 0:
                self.transitions[state] = {k: v/total for k, v in counts.items()}
        
        logger.debug(f"MarkovModel geçiş matrisi güncellendi: {len(self.transitions)} durum")
    
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
    
    def set_order(self, order):
        """
        Markov zinciri mertebesini ayarlar
        
        Args:
            order (int): Yeni mertebe değeri
        """
        if order < 1:
            logger.warning(f"MarkovModel: Geçersiz mertebe değeri: {order}, minimum 1 olmalı")
            return
        
        self.order = order
        # Geçiş matrisini sıfırla
        self.transitions = {}
        
        logger.info(f"MarkovModel mertebesi {order} olarak ayarlandı")