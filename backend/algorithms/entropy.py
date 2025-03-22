#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entropi Modeli
------------
Sonuçların entropisini (rastgeleliğini) analiz ederek tahmin yapar.
"""

import logging
import numpy as np
import math
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class EntropyModel(BaseAlgorithm):
    """
    Entropi Modeli
    
    Sonuçların entropisini (rastgeleliğini) analiz ederek tahmin yapar.
    Düşük entropi, belirli bir düzenin varlığını gösterir; yüksek entropi ise rastgeleliği gösterir.
    """
    
    def __init__(self, weight=1.0, db_manager=None, window_size=50):
        """
        EntropyModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            window_size (int): Entropi analizi pencere boyutu
        """
        super().__init__(name="Entropy Model", weight=weight)
        self.db_manager = db_manager
        self.window_size = window_size
        
        logger.info(f"Entropi Modeli başlatıldı: Pencere boyutu={window_size}")
    
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
            logger.warning("EntropyModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Son sonuçları al
        last_results = data.get('last_results', [])
        
        if len(last_results) < 10:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"EntropyModel: Yetersiz veri - {len(last_results)}/10")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Özel pencere boyutunu al veya varsayılanı kullan
        window_size = data.get('window_size', self.window_size)
        window_size = min(window_size, len(last_results))
        
        # Analiz edilecek sonuçları al
        window_results = last_results[-window_size:]
        
        # Sonuçların entropisini hesapla
        entropy = self._calculate_entropy(window_results)
        
        # Entropi düşükse (düzen varsa), desenleri analiz et
        if entropy < 1.0:  # Düşük entropi eşiği
            prediction, confidence, pattern_details = self._analyze_patterns(window_results)
        else:
            # Entropi yüksekse (rastgelelik varsa), teorik olasılıklara ve son eğilimlere bak
            prediction, confidence, pattern_details = self._analyze_random(window_results)
        
        # Detaylar
        details = {
            'entropy': entropy,
            'window_size': window_size,
            'is_random': entropy >= 1.0,
            'patterns': pattern_details
        }
        
        logger.debug(f"EntropyModel tahmini: {prediction}, Güven: {confidence:.4f}, Entropi: {entropy:.4f}")
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': details
        }
    
    def _calculate_entropy(self, results):
        """
        Shannon entropisini hesaplar
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            float: Entropi değeri
        """
        # Olasılık dağılımını hesapla
        counter = Counter(results)
        probs = [count / len(results) for count in counter.values()]
        
        # Shannon entropisini hesapla
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy
    
    def _analyze_patterns(self, results):
        """
        Düşük entropili sonuçlarda desen analizi yapar
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            tuple: (prediction, confidence, pattern_details)
        """
        # Son birkaç sonucu al
        last_3 = results[-3:]
        last_2 = results[-2:]
        
        # Tüm olası desenleri kontrol et (1, 2 ve 3 uzunluğunda)
        patterns = {}
        
        # 1 uzunluğundaki desenler (tek sonuç)
        for result in set(results):
            indices = [i for i, r in enumerate(results[:-1]) if r == result]
            next_results = [results[i+1] for i in indices]
            
            if next_results:
                counter = Counter(next_results)
                patterns[result] = {
                    'next': counter.most_common(1)[0][0],
                    'confidence': counter.most_common(1)[0][1] / len(next_results),
                    'count': len(indices)
                }
        
        # 2 uzunluğundaki desenler
        for i in range(len(results) - 2):
            pattern = ''.join(results[i:i+2])
            if pattern not in patterns:
                patterns[pattern] = {'count': 0, 'next_results': []}
            
            patterns[pattern]['count'] += 1
            patterns[pattern]['next_results'].append(results[i+2])
        
        # 3 uzunluğundaki desenler
        for i in range(len(results) - 3):
            pattern = ''.join(results[i:i+3])
            if pattern not in patterns:
                patterns[pattern] = {'count': 0, 'next_results': []}
            
            patterns[pattern]['count'] += 1
            if i+3 < len(results):
                patterns[pattern]['next_results'].append(results[i+3])
        
        # Desenlerdeki en yaygın sonucu hesapla
        for pattern, data in patterns.items():
            if pattern in [''.join(last_2), ''.join(last_3)] and 'next_results' in data and data['next_results']:
                counter = Counter(data['next_results'])
                data['next'] = counter.most_common(1)[0][0]
                data['next_confidence'] = counter.most_common(1)[0][1] / len(data['next_results'])
        
        # Son 3 sonucu desen olarak kullan
        pattern_3 = ''.join(last_3)
        if pattern_3 in patterns and 'next' in patterns[pattern_3]:
            prediction = patterns[pattern_3]['next']
            confidence = patterns[pattern_3]['next_confidence']
            logger.debug(f"3 uzunluklu desen bulundu: {pattern_3} -> {prediction}")
        else:
            # Son 2 sonucu desen olarak kullan
            pattern_2 = ''.join(last_2)
            if pattern_2 in patterns and 'next' in patterns[pattern_2]:
                prediction = patterns[pattern_2]['next']
                confidence = patterns[pattern_2]['next_confidence']
                logger.debug(f"2 uzunluklu desen bulundu: {pattern_2} -> {prediction}")
            else:
                # Son 1 sonucu desen olarak kullan
                pattern_1 = last_3[-1]
                if pattern_1 in patterns and 'next' in patterns[pattern_1]:
                    prediction = patterns[pattern_1]['next']
                    confidence = patterns[pattern_1]['next_confidence'] 
                    logger.debug(f"1 uzunluklu desen bulundu: {pattern_1} -> {prediction}")
                else:
                    # Desen bulunamadıysa, en sık sonucu kullan
                    counter = Counter(results)
                    prediction = counter.most_common(1)[0][0]
                    confidence = counter.most_common(1)[0][1] / len(results)
                    logger.debug(f"Desen bulunamadı, en sık sonuç kullanıldı: {prediction}")
        
        return prediction, confidence, patterns
    
    def _analyze_random(self, results):
        """
        Yüksek entropili (rastgele) sonuçlarda analiz yapar
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            tuple: (prediction, confidence, pattern_details)
        """
        # Teorik olasılıklar
        theoretical_probs = {'P': 0.4932, 'B': 0.5068, 'T': 0.0955}
        
        # Son sonuçların dağılımını hesapla
        counter = Counter(results)
        distribution = {k: counter.get(k, 0) / len(results) for k in ['P', 'B', 'T']}
        
        # Ortalamaya dönme eğilimini değerlendir
        probabilities = {}
        for outcome in ['P', 'B', 'T']:
            theoretical = theoretical_probs[outcome]
            actual = distribution.get(outcome, 0)
            
            # Sapma yönüne göre olasılık ayarla
            if actual > theoretical:
                # Normalden fazla görülmüş, azalma beklenir
                probabilities[outcome] = max(0.2, theoretical - (actual - theoretical) * 0.5)
            else:
                # Normalden az görülmüş, artış beklenir
                probabilities[outcome] = min(0.7, theoretical + (theoretical - actual) * 0.5)
        
        # Son 5 sonuçtaki eğilimi de dikkate al
        last_5 = results[-5:]
        recent_counter = Counter(last_5)
        
        # Eğilim ağırlığını hesapla
        for outcome in ['P', 'B', 'T']:
            recent_prob = recent_counter.get(outcome, 0) / len(last_5)
            # Son eğilimi %30 ağırlıkla dahil et
            probabilities[outcome] = probabilities[outcome] * 0.7 + recent_prob * 0.3
        
        # En olası sonucu belirle
        prediction = max(probabilities, key=probabilities.get)
        confidence = probabilities[prediction]
        
        # Detaylar
        pattern_details = {
            'theoretical': theoretical_probs,
            'observed': distribution,
            'adjusted': probabilities,
            'recent_trend': {k: recent_counter.get(k, 0) / len(last_5) for k in ['P', 'B', 'T']}
        }
        
        return prediction, confidence, pattern_details
    
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
        Entropi analizi pencere boyutunu ayarlar
        
        Args:
            window_size (int): Yeni pencere boyutu
        """
        if window_size < 10:
            logger.warning(f"EntropyModel: Geçersiz pencere boyutu: {window_size}, minimum 10 olmalı")
            return
        
        self.window_size = window_size
        logger.info(f"EntropyModel pencere boyutu {window_size} olarak ayarlandı")