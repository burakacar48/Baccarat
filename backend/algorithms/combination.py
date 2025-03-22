#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kombinasyon Analizi
----------------
Yukarıdaki algoritmaların birden fazlasını kullanarak ağırlıklı bir tahmin üretir.
"""

import logging
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class CombinationAnalysis(BaseAlgorithm):
    """
    Kombinasyon Analizi
    
    Farklı algoritmaların tahminlerini birleştirerek daha güçlü bir tahmin üretir.
    Her alt algoritma kendi tahminini yapar ve bu tahminler ağırlıklarına göre birleştirilir.
    """
    
    def __init__(self, weight=1.0, db_manager=None, algorithms=None):
        """
        CombinationAnalysis sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            algorithms (list, optional): Alt algoritmaların listesi
        """
        super().__init__(name="Combination Analysis", weight=weight)
        self.db_manager = db_manager
        self.algorithms = algorithms or []
        
        logger.info(f"Kombinasyon Analizi başlatıldı: {len(self.algorithms)} alt algoritma")
    
    def add_algorithm(self, algorithm, sub_weight=1.0):
        """
        Alt algoritma ekler
        
        Args:
            algorithm: Algoritma nesnesi (BaseAlgorithm türevi)
            sub_weight (float): Alt algoritmanın ağırlığı
        """
        self.algorithms.append({
            'algorithm': algorithm,
            'weight': sub_weight
        })
        
        logger.info(f"Alt algoritma eklendi: {algorithm.name}, Ağırlık: {sub_weight}")
    
    def predict(self, data):
        """
        Verilen verilere göre tahmin yapar
        
        Args:
            data (dict): Tahmin için gerekli veriler
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - confidence (float): Güven skoru (0-1 arası)
                - votes (dict): Her sonuç için toplam oylar
                - sub_predictions (list): Alt algoritmaların tahminleri
        """
        # Yeterli algoritma yoksa varsayılan tahmin
        if not self.algorithms:
            logger.warning("CombinationAnalysis: Alt algoritma yok, varsayılan tahmin yapılıyor")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Tüm algoritmalardan tahmin al
        predictions = []
        
        for algo_info in self.algorithms:
            algorithm = algo_info['algorithm']
            sub_weight = algo_info['weight']
            
            try:
                result = algorithm.predict(data)
                predictions.append({
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'weight': sub_weight
                })
                
                logger.debug(f"Alt algoritma tahmini: {algorithm.name} - {result['prediction']} ({result['confidence']:.4f})")
            except Exception as e:
                # Algoritma hatası, bu algoritmayı atla
                logger.error(f"Alt algoritma hatası ({algorithm.name}): {str(e)}")
                continue
        
        # Tahmin yoksa varsayılan değer
        if not predictions:
            logger.warning("CombinationAnalysis: Hiçbir alt algoritma tahmin üretemedi")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Ağırlıklı oylama
        votes = {'P': 0.0, 'B': 0.0, 'T': 0.0}
        
        for pred in predictions:
            # Ağırlık * güven skoru
            score = pred['weight'] * pred['confidence']
            votes[pred['prediction']] += score
        
        # En yüksek oyu alan tahmini bul
        total_votes = sum(votes.values())
        
        if total_votes > 0:
            # Normalize
            normalized_votes = {k: v/total_votes for k, v in votes.items()}
            
            # En yüksek oyu alan tahmin
            prediction = max(normalized_votes, key=normalized_votes.get)
            confidence = normalized_votes[prediction]
            
            # Alt tahminleri hazırla
            sub_predictions = [
                {
                    'algorithm': algo_info['algorithm'].name,
                    'prediction': pred['prediction'],
                    'confidence': pred['confidence']
                }
                for algo_info, pred in zip(self.algorithms, predictions)
            ]
            
            logger.info(f"CombinationAnalysis tahmini: {prediction}, Güven: {confidence:.4f}, Alt tahmin sayısı: {len(predictions)}")
            return {
                'prediction': prediction,
                'confidence': confidence,
                'votes': normalized_votes,
                'sub_predictions': sub_predictions
            }
        
        # Oy yoksa varsayılan değer
        logger.warning("CombinationAnalysis: Geçerli oy yok")
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
    
    def remove_algorithm(self, algorithm_name):
        """
        İsme göre alt algoritmayı kaldırır
        
        Args:
            algorithm_name (str): Kaldırılacak algoritmanın adı
            
        Returns:
            bool: Başarılı ise True, değilse False
        """
        initial_count = len(self.algorithms)
        
        self.algorithms = [
            algo for algo in self.algorithms 
            if algo['algorithm'].name != algorithm_name
        ]
        
        if len(self.algorithms) < initial_count:
            logger.info(f"Alt algoritma kaldırıldı: {algorithm_name}")
            return True
        
        logger.warning(f"Alt algoritma bulunamadı: {algorithm_name}")
        return False
    
    def update_weight(self, algorithm_name, new_weight):
        """
        Alt algoritmanın ağırlığını günceller
        
        Args:
            algorithm_name (str): Güncellenecek algoritmanın adı
            new_weight (float): Yeni ağırlık değeri
            
        Returns:
            bool: Başarılı ise True, değilse False
        """
        for algo in self.algorithms:
            if algo['algorithm'].name == algorithm_name:
                old_weight = algo['weight']
                algo['weight'] = max(0.1, new_weight)  # Minimum 0.1 ağırlık
                
                logger.info(f"Alt algoritma ağırlığı güncellendi: {algorithm_name}, {old_weight:.2f} -> {algo['weight']:.2f}")
                return True
        
        logger.warning(f"Alt algoritma bulunamadı: {algorithm_name}")
        return False