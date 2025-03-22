#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sonuç Birleştirici
-----------------
Farklı algoritmalardan gelen tahminleri birleştirip nihai tahmin üretir.
"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)

class ResultAggregator:
    """
    Farklı algoritmaların tahminlerini birleştiren sınıf
    
    Algoritmaların ağırlıkları, güven skorları ve doğruluk oranlarını dikkate alarak
    en iyi tahmin sonucunu üretir.
    """
    
    def __init__(self, strategies=None):
        """
        ResultAggregator sınıfını başlatır
        
        Args:
            strategies (list, optional): Kullanılacak birleştirme stratejileri listesi
        """
        self.strategies = strategies or ["weighted_voting", "confidence_adjusted", "accuracy_based"]
        self.current_strategy = "weighted_voting"
        logger.info(f"Sonuç birleştirici başlatıldı: Stratejiler={self.strategies}")
    
    def set_strategy(self, strategy):
        """
        Kullanılacak birleştirme stratejisini ayarlar
        
        Args:
            strategy (str): Strateji adı
        """
        if strategy in self.strategies:
            self.current_strategy = strategy
            logger.info(f"Birleştirme stratejisi değiştirildi: {strategy}")
        else:
            logger.warning(f"Geçersiz strateji: {strategy}, mevcut stratejiler: {self.strategies}")
    
    def aggregate(self, algorithm_predictions):
        """
        Tahminleri birleştirir ve nihai tahmin üretir
        
        Args:
            algorithm_predictions (list): Algoritma tahminlerinin listesi
                Her tahmin şu formatta bir dict olmalı:
                {
                    'algorithm': str,  # Algoritma adı
                    'prediction': str,  # Tahmin (P/B/T)
                    'confidence': float,  # Güven skoru (0-1)
                    'weight': float  # Algoritma ağırlığı
                }
        
        Returns:
            dict: Nihai tahmin sonucu
                {
                    'prediction': str,  # Nihai tahmin (P/B/T)
                    'confidence': float,  # Nihai güven skoru
                    'details': dict  # Her sonucun ayrıntılı olasılıkları
                }
        """
        if not algorithm_predictions:
            logger.warning("Birleştirilecek tahmin yok. Varsayılan tahmin döndürülüyor.")
            return {
                'prediction': 'B',
                'confidence': 0.5,
                'details': {'P': 0.0, 'B': 0.5, 'T': 0.0}
            }
        
        # Seçilen stratejiye göre tahminleri birleştir
        if self.current_strategy == "weighted_voting":
            result = self._weighted_voting(algorithm_predictions)
        elif self.current_strategy == "confidence_adjusted":
            result = self._confidence_adjusted(algorithm_predictions)
        elif self.current_strategy == "accuracy_based":
            result = self._accuracy_based(algorithm_predictions)
        else:
            # Varsayılan olarak ağırlıklı oylama kullan
            result = self._weighted_voting(algorithm_predictions)
        
        # Sonuçları logla
        logger.info(f"Nihai tahmin: {result['prediction']}, Güven: {result['confidence']:.4f}")
        return result
    
    def _weighted_voting(self, algorithm_predictions):
        """
        Ağırlıklı oylama stratejisi
        
        Her algoritmanın tahminine kendi ağırlığı kadar değer verir.
        
        Args:
            algorithm_predictions (list): Algoritma tahminleri
        
        Returns:
            dict: Nihai tahmin sonucu
        """
        # Her sonuç için oy sayısını izle
        votes = {'P': 0.0, 'B': 0.0, 'T': 0.0}
        
        # Tahminleri topla
        for pred in algorithm_predictions:
            votes[pred['prediction']] += pred['weight']
        
        # En çok oy alan sonucu bul
        prediction = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[prediction] / total_votes if total_votes > 0 else 0.5
        
        # Detaylı olasılıkları hesapla
        details = {k: v / total_votes for k, v in votes.items()}
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': details
        }
    
    def _confidence_adjusted(self, algorithm_predictions):
        """
        Güven skoru ayarlı strateji
        
        Her algoritmanın tahminine kendi güven skoru ve ağırlığı kadar değer verir.
        
        Args:
            algorithm_predictions (list): Algoritma tahminleri
        
        Returns:
            dict: Nihai tahmin sonucu
        """
        # Her sonuç için ağırlıklı güven skorlarını izle
        confidence_scores = {'P': 0.0, 'B': 0.0, 'T': 0.0}
        weights = {'P': 0.0, 'B': 0.0, 'T': 0.0}
        
        # Ağırlıklı güven skorlarını topla
        for pred in algorithm_predictions:
            prediction = pred['prediction']
            confidence = pred['confidence']
            weight = pred['weight']
            
            confidence_scores[prediction] += confidence * weight
            weights[prediction] += weight
        
        # Ağırlıklı ortalama güven skorlarını hesapla
        avg_confidence = {}
        for k in confidence_scores:
            if weights[k] > 0:
                avg_confidence[k] = confidence_scores[k] / weights[k]
            else:
                avg_confidence[k] = 0.0
        
        # En yüksek güven skoruna sahip sonucu bul
        prediction = max(avg_confidence, key=avg_confidence.get)
        confidence = avg_confidence[prediction]
        
        # Toplam değer
        total_value = sum(confidence_scores.values())
        
        # Detaylı olasılıkları hesapla
        if total_value > 0:
            details = {k: v / total_value for k, v in confidence_scores.items()}
        else:
            details = {k: 1/3 for k in confidence_scores}
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': details
        }
    
    def _accuracy_based(self, algorithm_predictions):
        """
        Doğruluk oranı tabanlı strateji
        
        Algoritmaların geçmiş doğruluk oranlarına dayalı bir tahmin üretir.
        
        Args:
            algorithm_predictions (list): Algoritma tahminleri
        
        Returns:
            dict: Nihai tahmin sonucu
        """
        # Her sonuç için ağırlıklı doğruluk skorlarını izle
        scores = {'P': 0.0, 'B': 0.0, 'T': 0.0}
        total_score = 0.0
        
        # Algoritmaların tahminlerini işle
        for pred in algorithm_predictions:
            prediction = pred['prediction']
            confidence = pred['confidence']
            weight = pred['weight']
            
            # Tahmine ağırlık ve güven skoru uygula
            score = weight * confidence
            scores[prediction] += score
            total_score += score
        
        # En yüksek skora sahip sonucu bul
        prediction = max(scores, key=scores.get)
        
        # Normalize edilmiş skorları hesapla
        if total_score > 0:
            confidence = scores[prediction] / total_score
            details = {k: v / total_score for k, v in scores.items()}
        else:
            confidence = 1/3
            details = {k: 1/3 for k in scores}
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': details
        }
    
    def get_algorithm_contributions(self, algorithm_predictions, final_prediction):
        """
        Her algoritmanın nihai tahmine katkısını hesaplar
        
        Args:
            algorithm_predictions (list): Algoritma tahminleri
            final_prediction (str): Nihai tahmin
        
        Returns:
            dict: Her algoritmanın katkı yüzdesi
        """
        contributions = {}
        total_contribution = 0.0
        
        for pred in algorithm_predictions:
            algorithm = pred['algorithm']
            prediction = pred['prediction']
            weight = pred['weight']
            confidence = pred['confidence']
            
            # Eğer algoritmanın tahmini nihai tahmin ile aynıysa katkı hesapla
            if prediction == final_prediction:
                contribution = weight * confidence
                contributions[algorithm] = contribution
                total_contribution += contribution
        
        # Katkıları normalize et
        if total_contribution > 0:
            normalized_contributions = {k: v / total_contribution for k, v in contributions.items()}
        else:
            normalized_contributions = {k: 0.0 for k in contributions}
        
        return normalized_contributions