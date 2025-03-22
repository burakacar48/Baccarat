#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte Carlo Simülasyon Modeli
-------------------------
Mevcut verilere dayalı olarak çok sayıda simülasyon çalıştırarak olası sonuçları tahmin eder.
"""

import logging
import random
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class MonteCarloSimulation(BaseAlgorithm):
    """
    Monte Carlo Simülasyon Modeli
    
    Mevcut verilere dayalı olarak çok sayıda simülasyon çalıştırarak olası sonuçları tahmin eder.
    Geçiş olasılıklarını hesaplar ve bu olasılıklara dayalı rastgele simülasyonlar yapar.
    """
    
    def __init__(self, weight=1.0, db_manager=None, simulations=1000):
        """
        MonteCarloSimulation sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            simulations (int): Çalıştırılacak Monte Carlo simülasyon sayısı
        """
        super().__init__(name="Monte Carlo Simulation", weight=weight)
        self.db_manager = db_manager
        self.simulations = simulations
        
        logger.info(f"Monte Carlo Simülasyon Modeli başlatıldı: Simülasyon sayısı={simulations}")
    
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
            logger.warning("MonteCarloSimulation: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Son N sonucu al
        last_results = data.get('last_results', [])
        
        if len(last_results) < 10:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"MonteCarloSimulation: Yetersiz veri - {len(last_results)}/10")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Geçiş olasılıklarını hesapla
        transitions = self._calculate_transition_probabilities(last_results)
        
        # Mevcut durum (son sonuç)
        current_state = last_results[-1]
        
        # Simülasyonları çalıştır
        simulation_results = []
        
        for _ in range(self.simulations):
            next_state = self._next_state(current_state, transitions)
            simulation_results.append(next_state)
        
        # Sonuçları say
        counter = Counter(simulation_results)
        total = len(simulation_results)
        
        # En olası sonucu bul
        if total > 0:
            sorted_results = counter.most_common()
            prediction = sorted_results[0][0]
            confidence = sorted_results[0][1] / total
            
            # Detaylar
            details = {
                'simulations': self.simulations,
                'distribution': {k: v/total for k, v in counter.items()},
                'transitions': {k: dict(v) for k, v in transitions.items()}
            }
            
            logger.debug(f"MonteCarloSimulation tahmini: {prediction}, Güven: {confidence:.4f}")
            return {
                'prediction': prediction,
                'confidence': confidence,
                'details': details
            }
        
        # Simülasyon başarısız olduysa varsayılan tahmin
        logger.warning("MonteCarloSimulation: Simülasyon sonuçları alınamadı")
        return {
            'prediction': 'B',
            'confidence': 0.5
        }
    
    def _calculate_transition_probabilities(self, results):
        """
        1. mertebe Markov geçiş olasılıklarını hesaplar
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            dict: Geçiş olasılıkları matrisi
        """
        transitions = {
            'P': {'P': 0, 'B': 0, 'T': 0},
            'B': {'P': 0, 'B': 0, 'T': 0},
            'T': {'P': 0, 'B': 0, 'T': 0}
        }
        
        # Geçişleri say
        for i in range(len(results) - 1):
            current = results[i]
            next_state = results[i+1]
            transitions[current][next_state] += 1
        
        # Olasılıklara dönüştür
        for state in transitions:
            total = sum(transitions[state].values())
            if total > 0:
                transitions[state] = {k: v/total for k, v in transitions[state].items()}
            else:
                # Eğer hiç geçiş yoksa, eşit olasılık ver
                transitions[state] = {'P': 0.33, 'B': 0.34, 'T': 0.33}
        
        return transitions
    
    def _next_state(self, current_state, transitions):
        """
        Geçiş olasılıklarına göre sonraki durumu belirler
        
        Args:
            current_state (str): Mevcut durum
            transitions (dict): Geçiş olasılıkları matrisi
            
        Returns:
            str: Bir sonraki durum
        """
        probs = transitions[current_state]
        outcomes = list(probs.keys())
        probabilities = list(probs.values())
        
        # Rastgele sonraki durumu seç
        return random.choices(outcomes, weights=probabilities, k=1)[0]
    
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
    
    def set_simulations(self, simulations):
        """
        Simülasyon sayısını ayarlar
        
        Args:
            simulations (int): Yeni simülasyon sayısı
        """
        if simulations < 100:
            logger.warning(f"MonteCarloSimulation: Geçersiz simülasyon sayısı: {simulations}, minimum 100 olmalı")
            return
        
        self.simulations = simulations
        logger.info(f"MonteCarloSimulation simülasyon sayısı {simulations} olarak ayarlandı")