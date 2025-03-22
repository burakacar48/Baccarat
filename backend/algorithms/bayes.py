#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bayes Temelli Adaptif Model
-------------------------
Her yeni sonuç geldiğinde kendini güncelleyen adaptif bir model kullanır.
"""

import logging
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class BayesModel(BaseAlgorithm):
    """
    Bayes Temelli Adaptif Model
    
    Her yeni sonuç geldiğinde kendini güncelleyen adaptif bir model kullanır.
    Koşullu olasılıkları ve prior olasılıkları güncelleyerek tahmin yapar.
    """
    
    def __init__(self, weight=1.0, db_manager=None):
        """
        BayesModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
        """
        super().__init__(name="Bayes Model", weight=weight)
        self.db_manager = db_manager
        
        # Başlangıç olasılıkları (prior)
        self.priors = {'P': 0.4932, 'B': 0.5068, 'T': 0.0955}
        
        # Koşullu olasılıklar (başlangıçta eşit)
        self.conditionals = {
            'P': {'P': 0.33, 'B': 0.33, 'T': 0.33},
            'B': {'P': 0.33, 'B': 0.33, 'T': 0.33},
            'T': {'P': 0.33, 'B': 0.33, 'T': 0.33}
        }
        
        logger.info("Bayes Model başlatıldı")
    
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
            logger.warning("BayesModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',  # Varsayılan olarak Banker'ı döndür
                'confidence': self.priors['B']
            }
        
        # Son sonucu al
        last_results = data['last_results']
        last_result = last_results[-1]
        
        # Koşullu olasılıkları kullanarak tahmin yap
        posteriors = {}
        for outcome in ['P', 'B', 'T']:
            # P(outcome | last_result) orantılı P(last_result | outcome) * P(outcome)
            posteriors[outcome] = self.conditionals[last_result].get(outcome, 0.33) * self.priors[outcome]
        
        # Normalize et
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}
        
        # En olası sonucu bul
        prediction = max(posteriors, key=posteriors.get)
        confidence = posteriors[prediction]
        
        logger.debug(f"BayesModel tahmini: {prediction}, Güven: {confidence:.4f}")
        return {
            'prediction': prediction,
            'confidence': confidence
        }
    
    def update_model(self, last_result, current_result):
        """
        Modeli günceller
        
        Args:
            last_result (str): Önceki sonuç
            current_result (str): Şu anki sonuç
        """
        # Koşullu olasılıkları güncelle
        # P(current | last) olasılığını arttır
        alpha = 0.1  # Öğrenme oranı
        
        # Mevcut değeri al
        current_prob = self.conditionals[last_result].get(current_result, 0.33)
        
        # Güncelle
        self.conditionals[last_result][current_result] = current_prob * (1-alpha) + alpha
        
        # Diğer olasılıkları normalleştir
        others = [o for o in ['P', 'B', 'T'] if o != current_result]
        for other in others:
            self.conditionals[last_result][other] *= (1-alpha)
        
        # Prior olasılıkları da güncelle
        self.priors[current_result] = self.priors[current_result] * 0.99 + 0.01
        for other in others:
            self.priors[other] *= 0.99
            
        logger.debug(f"BayesModel güncellendi: {last_result} -> {current_result}")
    
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