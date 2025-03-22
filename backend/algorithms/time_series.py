#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zaman Serisi Modeli
----------------
Baccarat sonuçlarını zaman serisi olarak analiz eder ve gelecekteki değerleri tahmin eder.
"""

import logging
import numpy as np
from collections import Counter
from datetime import datetime
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class TimeSeriesModel(BaseAlgorithm):
    """
    Zaman Serisi Modeli
    
    Baccarat sonuçlarını zaman serisi olarak analiz eder ve gelecekteki değerleri
    tahmin etmek için hareketli ortalama, üstel düzeltme gibi teknikleri kullanır.
    """
    
    def __init__(self, weight=1.0, db_manager=None, window_size=10):
        """
        TimeSeriesModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            window_size (int): Hareketli ortalama pencere boyutu
        """
        super().__init__(name="Time Series Model", weight=weight)
        self.db_manager = db_manager
        self.window_size = window_size
        
        # Statsmodels'in yüklü olup olmadığını kontrol et
        try:
            import statsmodels.api as sm
            self.statsmodels_available = True
        except ImportError:
            self.statsmodels_available = False
            logger.warning("TimeSeriesModel: statsmodels kütüphanesi bulunamadı, basit zaman serisi analizi kullanılacak")
        
        logger.info(f"Zaman Serisi Modeli başlatıldı: Pencere boyutu={window_size}")
    
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
            logger.warning("TimeSeriesModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Son sonuçları al
        last_results = data.get('last_results', [])
        
        if len(last_results) < 20:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"TimeSeriesModel: Yetersiz veri - {len(last_results)}/20")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Statsmodels yoksa basit zaman serisi analizi yap
        if not self.statsmodels_available:
            return self._simple_time_series(last_results)
        
        # Özel pencere boyutunu al veya varsayılanı kullan
        window_size = data.get('window_size', self.window_size)
        
        try:
            # Sonuçları sayısal değerlere dönüştür
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = [result_map.get(r, 1) for r in last_results]
            
            # Eksik verileri ara değer ile doldur
            import statsmodels.api as sm
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Son 30 sonucu al (veya mevcut tüm sonuçları)
            recent_results = numerical_results[-min(30, len(numerical_results)):]
            
            # Üstel düzgünleştirme modeli (Holt-Winters)
            model = ExponentialSmoothing(
                np.array(recent_results),
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            model_fit = model.fit()
            
            # Bir sonraki değeri tahmin et
            forecast = model_fit.forecast(1)[0]
            
            # En yakın sınıfa yuvarla
            nearest_class = int(round(forecast))
            
            # Sınır kontrolü
            if nearest_class < 0:
                nearest_class = 0
            elif nearest_class > 2:
                nearest_class = 2
            
            # Sayısal değerden sembol değerine dönüştür
            reverse_map = {0: 'P', 1: 'B', 2: 'T'}
            prediction = reverse_map[nearest_class]
            
            # Güven değerini hesapla (son tahminlerin doğruluğuna bağlı)
            confidence = 0.5
            
            # Son tahminleri kontrol et
            if len(recent_results) > 5:
                accuracy = 0
                for i in range(1, 5):
                    test_data = recent_results[:-i]
                    test_model = ExponentialSmoothing(
                        np.array(test_data),
                        trend='add',
                        seasonal=None,
                        initialization_method='estimated'
                    )
                    test_fit = test_model.fit()
                    test_forecast = int(round(test_fit.forecast(1)[0]))
                    if test_forecast == recent_results[-i]:
                        accuracy += 1
                
                # Doğruluğa göre güven skorunu ayarla
                confidence = 0.4 + (accuracy / 10)  # 0.4 - 0.8 arası
            
            # Detaylar
            details = {
                'forecast': float(forecast),
                'model': 'exponential_smoothing',
                'recent_values': [int(x) for x in recent_results[-5:]]
            }
            
            logger.debug(f"TimeSeriesModel tahmini: {prediction}, Güven: {confidence:.4f}, Tahmin: {forecast:.4f}")
            return {
                'prediction': prediction,
                'confidence': confidence,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"TimeSeriesModel hatası: {str(e)}")
        
        # Hata durumunda basit zaman serisi analizi yap
        return self._simple_time_series(last_results)
    
    def _simple_time_series(self, results):
        """
        Basit bir zaman serisi analizi yapar (statsmodels olmadığında kullanılır)
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            dict: Tahmin sonucu
        """
        # Sonuçları sayısal değerlere dönüştür
        result_map = {'P': 0, 'B': 1, 'T': 2}
        numerical_results = [result_map.get(r, 1) for r in results]
        
        # Son 20 sonucu al
        recent_results = numerical_results[-min(20, len(numerical_results)):]
        
        # Hareketli ortalama hesapla
        ma_window = min(5, len(recent_results))
        moving_average = np.convolve(recent_results, np.ones(ma_window)/ma_window, mode='valid')
        
        # Son hareketli ortalama değerini al
        last_ma = moving_average[-1]
        
        # En yakın sınıfa yuvarla
        nearest_class = int(round(last_ma))
        
        # Sınır kontrolü
        if nearest_class < 0:
            nearest_class = 0
        elif nearest_class > 2:
            nearest_class = 2
        
        # Sayısal değerden sembol değerine dönüştür
        reverse_map = {0: 'P', 1: 'B', 2: 'T'}
        prediction = reverse_map[nearest_class]
        
        # Son 5 sonuç içindeki en yaygın sınıfı bul
        recent_5 = recent_results[-5:]
        counter = Counter(recent_5)
        most_common = counter.most_common(1)[0][0]
        
        # Hareketli ortalama ve en yaygın sınıf aynıysa güveni artır
        confidence = 0.5
        if nearest_class == most_common:
            confidence += 0.1
        
        # Yöne göre güveni ayarla
        if len(moving_average) > 1:
            trend = moving_average[-1] - moving_average[-2]
            if abs(trend) > 0.1:
                confidence += 0.05
        
        logger.debug(f"TimeSeriesModel (basit) tahmini: {prediction}, Güven: {confidence:.4f}, MA: {last_ma:.4f}")
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': {
                'moving_average': float(last_ma),
                'recent_values': [int(x) for x in recent_results[-5:]],
                'ma_window': ma_window
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
        Hareketli ortalama pencere boyutunu ayarlar
        
        Args:
            window_size (int): Yeni pencere boyutu
        """
        if window_size < 2:
            logger.warning(f"TimeSeriesModel: Geçersiz pencere boyutu: {window_size}, minimum 2 olmalı")
            return
        
        self.window_size = window_size
        logger.info(f"TimeSeriesModel pencere boyutu {window_size} olarak ayarlandı")