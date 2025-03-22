#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regresyon Modeli
--------------
Geçmiş sonuçların analizi ile gelecek sonuçları tahmin etmek için regresyon analizi kullanır.
"""

import logging
import numpy as np
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class RegressionModel(BaseAlgorithm):
    """
    Regresyon Modeli
    
    Geçmiş sonuçların analizi ile gelecek sonuçları tahmin etmek için regresyon analizi kullanır.
    Sonuçları sayısal değerlere dönüştürerek eğilimi analiz eder.
    """
    
    def __init__(self, weight=1.0, db_manager=None, window_size=50):
        """
        RegressionModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            window_size (int): Regresyon analizi pencere boyutu
        """
        super().__init__(name="Regression Model", weight=weight)
        self.db_manager = db_manager
        self.window_size = window_size
        
        # Sklearn'in yüklü olup olmadığını kontrol et
        try:
            from sklearn.linear_model import LogisticRegression
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            logger.warning("RegressionModel: sklearn kütüphanesi bulunamadı, basit regresyon kullanılacak")
        
        logger.info(f"Regresyon Modeli başlatıldı: Pencere boyutu={window_size}")
    
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
            logger.warning("RegressionModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Son sonuçları al
        last_results = data.get('last_results', [])
        
        if len(last_results) < 20:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"RegressionModel: Yetersiz veri - {len(last_results)}/20")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Sklearn yoksa basit regresyon yap
        if not self.sklearn_available:
            return self._simple_regression(last_results)
        
        # Özel pencere boyutunu al veya varsayılanı kullan
        window_size = data.get('window_size', self.window_size)
        
        try:
            # Sonuçları sayısal değerlere dönüştür
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = [result_map.get(r, 1) for r in last_results]
            
            # Özellikleri oluştur (son 5 sonuç ile tahmin yap)
            X, y = [], []
            
            for i in range(len(numerical_results) - 5):
                X.append(numerical_results[i:i+5])
                y.append(numerical_results[i+5])
            
            # Yeterli veri yoksa basit regresyon yap
            if len(X) < 10:
                logger.warning(f"RegressionModel: Eğitim için yetersiz veri - {len(X)}/10")
                return self._simple_regression(last_results)
            
            # Lojistik regresyon modeli
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            # Verileri ölçekle
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Modeli eğit
            model = LogisticRegression(max_iter=1000, multi_class='multinomial')
            model.fit(X_scaled, y)
            
            # Son 5 sonuç ile tahmin yap
            last_5 = numerical_results[-5:]
            last_5_scaled = scaler.transform([last_5])
            
            # Olasılıkları hesapla
            probabilities = model.predict_proba(last_5_scaled)[0]
            
            # En yüksek olasılıklı sınıfı bul
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            
            # Sayısal değerden sembol değerine dönüştür
            reverse_map = {0: 'P', 1: 'B', 2: 'T'}
            prediction = reverse_map[prediction_idx]
            
            # Detaylar
            details = {
                'probabilities': {
                    'P': float(probabilities[0]),
                    'B': float(probabilities[1]),
                    'T': float(probabilities[2])
                },
                'model': 'logistic_regression',
                'features': [int(x) for x in last_5]
            }
            
            logger.debug(f"RegressionModel tahmini: {prediction}, Güven: {confidence:.4f}")
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'details': details
            }
            
        except Exception as e:
            logger.error(f"RegressionModel hatası: {str(e)}")
        
        # Hata durumunda basit regresyon yap
        return self._simple_regression(last_results)
    
    def _simple_regression(self, results):
        """
        Basit bir regresyon analizi yapar (sklearn olmadığında kullanılır)
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            dict: Tahmin sonucu
        """
        # Sonuçları sayısal değerlere dönüştür
        result_map = {'P': 0, 'B': 1, 'T': 2}
        numerical_results = [result_map.get(r, 1) for r in results]
        
        # Son 10 sonucu al
        recent_results = numerical_results[-10:]
        
        # Eğilim hesapla (basit lineer regresyon)
        x = np.arange(len(recent_results))
        y = np.array(recent_results)
        
        # y = mx + b formülünü kullanarak eğilimi hesapla
        m, b = np.polyfit(x, y, 1)
        
        # Bir sonraki değeri tahmin et
        next_x = len(recent_results)
        predicted_value = m * next_x + b
        
        # En yakın sınıfa yuvarla
        nearest_class = int(round(predicted_value))
        
        # Sınır kontrolü
        if nearest_class < 0:
            nearest_class = 0
        elif nearest_class > 2:
            nearest_class = 2
        
        # Sayısal değerden sembol değerine dönüştür
        reverse_map = {0: 'P', 1: 'B', 2: 'T'}
        prediction = reverse_map[nearest_class]
        
        # Güven değerini hesapla (eğilim gücüne ve son sonuçlara bağlı)
        confidence = 0.5
        
        # Eğilim güçlü ise güveni artır
        if abs(m) > 0.1:
            confidence += 0.1
        
        # Son sonuçlarda tahmin edilen sınıf baskınsa güveni artır
        recent_class_count = recent_results.count(nearest_class)
        confidence += 0.1 * (recent_class_count / len(recent_results))
        
        # Güveni 0.3-0.7 aralığında sınırla
        confidence = max(0.3, min(0.7, confidence))
        
        logger.debug(f"RegressionModel (basit) tahmini: {prediction}, Güven: {confidence:.4f}, Eğilim: {m:.4f}")
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': {
                'trend': float(m),
                'intercept': float(b),
                'predicted_value': float(predicted_value)
            }
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
        Regresyon analizi pencere boyutunu ayarlar
        
        Args:
            window_size (int): Yeni pencere boyutu
        """
        if window_size < 10:
            logger.warning(f"RegressionModel: Geçersiz pencere boyutu: {window_size}, minimum 10 olmalı")
            return
        
        self.window_size = window_size
        logger.info(f"RegressionModel pencere boyutu {window_size} olarak ayarlandı")