#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kümeleme ve Anomali Tespiti (Clustering and Anomaly Detection)
--------------------------------------------------------
Baccarat sonuçlarında normal ve anormal dağılımları tespit eder.
"""

import logging
import numpy as np
from collections import Counter
from backend.algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)

class ClusteringModel(BaseAlgorithm):
    """
    Kümeleme ve Anomali Tespiti Modeli
    
    Baccarat sonuçlarında normal ve anormal dağılımları tespit eder.
    K-means gibi kümeleme algoritmaları kullanarak benzer sonuç örüntülerini gruplar.
    """
    
    def __init__(self, weight=1.0, db_manager=None, n_clusters=3, window_size=50):
        """
        ClusteringModel sınıfını başlatır
        
        Args:
            weight (float): Algoritmanın ağırlığı
            db_manager: Veritabanı yönetici nesnesi
            n_clusters (int): Küme sayısı
            window_size (int): Analiz edilecek pencere boyutu
        """
        super().__init__(name="Clustering Model", weight=weight)
        self.db_manager = db_manager
        self.n_clusters = n_clusters
        self.window_size = window_size
        
        # Sklearn'in yüklü olup olmadığını kontrol et
        try:
            from sklearn.cluster import KMeans
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            logger.warning("ClusteringModel: sklearn kütüphanesi bulunamadı, basit kümeleme kullanılacak")
        
        logger.info(f"Kümeleme Modeli başlatıldı: Küme sayısı={n_clusters}, Pencere boyutu={window_size}")
    
    def predict(self, data):
        """
        Verilen verilere göre tahmin yapar
        
        Args:
            data (dict): Tahmin için gerekli veriler
                - last_results (list): Son oyun sonuçları listesi
                - n_clusters (int, optional): Özel küme sayısı
                - window_size (int, optional): Özel pencere boyutu
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - confidence (float): Güven skoru (0-1 arası)
                - details (dict): Ek detaylar
        """
        # Veri doğrulama
        if 'last_results' not in data or not data['last_results']:
            logger.warning("ClusteringModel: Geçersiz veri - son sonuçlar bulunamadı")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Son sonuçları al
        last_results = data.get('last_results', [])
        
        if len(last_results) < 20:
            # Yeterli veri yoksa varsayılan tahmin
            logger.warning(f"ClusteringModel: Yetersiz veri - {len(last_results)}/20")
            return {
                'prediction': 'B',
                'confidence': 0.5
            }
        
        # Sklearn yoksa basit kümeleme yap
        if not self.sklearn_available:
            return self._simple_clustering(last_results)
        
        # Özel parametreleri al veya varsayılanları kullan
        n_clusters = data.get('n_clusters', self.n_clusters)
        window_size = data.get('window_size', self.window_size)
        
        try:
            # Sonuçları sayısal değerlere dönüştür
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = [result_map.get(r, 1) for r in last_results]
            
            # Kümeden sonraki olası değerleri bul
            # Pencere boyutuna göre sonuçları grupla
            features = []
            next_values = []
            
            window_size = min(10, len(numerical_results) - 1)
            for i in range(len(numerical_results) - window_size):
                window = numerical_results[i:i+window_size]
                next_val = numerical_results[i+window_size]
                
                # Özellik vektörü: her değerin sayısı
                feature = [
                    window.count(0)/window_size,  # P oranı
                    window.count(1)/window_size,  # B oranı
                    window.count(2)/window_size   # T oranı
                ]
                features.append(feature)
                next_values.append(next_val)
            
            # Kümeleme için yeterli veri var mı?
            if len(features) < n_clusters * 2:
                logger.warning(f"ClusteringModel: Kümeleme için yetersiz veri - {len(features)}/{n_clusters * 2}")
                return self._simple_clustering(last_results)
            
            # K-means kümeleme
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            clusters = kmeans.fit_predict(features)
            
            # Son pencereyi al
            last_window = numerical_results[-window_size:]
            last_feature = [
                last_window.count(0)/window_size,
                last_window.count(1)/window_size,
                last_window.count(2)/window_size
            ]
            
            # Son pencere hangi kümede?
            last_cluster = kmeans.predict([last_feature])[0]
            
            # Bu kümedeki elemanların sonraki değerlerini topla
            cluster_indices = [i for i, c in enumerate(clusters) if c == last_cluster]
            cluster_next_values = [next_values[i] for i in cluster_indices]
            
            # En sık sonucu bul
            counter = Counter(cluster_next_values)
            
            if counter:
                most_common = counter.most_common(1)[0]
                next_value = most_common[0]
                confidence = most_common[1] / len(cluster_next_values)
                
                # Sayısal değerden sembol değerine dönüştür
                reverse_map = {0: 'P', 1: 'B', 2: 'T'}
                prediction = reverse_map[next_value]
                
                # Detaylar
                details = {
                    'n_clusters': n_clusters,
                    'window_size': window_size,
                    'last_cluster': int(last_cluster),
                    'cluster_size': len(cluster_indices),
                    'distribution': {reverse_map[k]: v/len(cluster_next_values) for k, v in counter.items()}
                }
                
                logger.debug(f"ClusteringModel tahmini: {prediction}, Güven: {confidence:.4f}, Küme: {last_cluster}")
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'details': details
                }
            
        except Exception as e:
            logger.error(f"ClusteringModel hatası: {str(e)}")
        
        # Hata durumunda basit kümeleme yap
        return self._simple_clustering(last_results)
    
    def _simple_clustering(self, results):
        """
        Basit bir kümeleme analizi yapar (sklearn olmadığında kullanılır)
        
        Args:
            results (list): Sonuç listesi
            
        Returns:
            dict: Tahmin sonucu
        """
        # Son 10 sonucu al
        recent_results = results[-10:]
        counter = Counter(recent_results)
        
        # Oranları hesapla
        total = len(recent_results)
        distribution = {k: v/total for k, v in counter.items()}
        
        # En sık sonucu bul
        prediction = counter.most_common(1)[0][0]
        confidence = distribution[prediction]
        
        # Son 3 sonuçtaki trendi kontrol et
        last_3 = results[-3:]
        trend_counter = Counter(last_3)
        trend_prediction = trend_counter.most_common(1)[0][0]
        
        # Eğer son trendde farklı bir tahmin varsa ve güveni yüksekse, onu kullan
        if trend_prediction != prediction and trend_counter[trend_prediction] >= 2:
            prediction = trend_prediction
            confidence = trend_counter[trend_prediction] / 3
        
        logger.debug(f"ClusteringModel (basit) tahmini: {prediction}, Güven: {confidence:.4f}")
        return {
            'prediction': prediction,
            'confidence': confidence,
            'details': {
                'distribution': distribution,
                'trend': dict(trend_counter)
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
    
    def set_n_clusters(self, n_clusters):
        """
        Küme sayısını ayarlar
        
        Args:
            n_clusters (int): Yeni küme sayısı
        """
        if n_clusters < 2:
            logger.warning(f"ClusteringModel: Geçersiz küme sayısı: {n_clusters}, minimum 2 olmalı")
            return
        
        self.n_clusters = n_clusters
        logger.info(f"ClusteringModel küme sayısı {n_clusters} olarak ayarlandı")
    
    def set_window_size(self, window_size):
        """
        Pencere boyutunu ayarlar
        
        Args:
            window_size (int): Yeni pencere boyutu
        """
        if window_size < 5:
            logger.warning(f"ClusteringModel: Geçersiz pencere boyutu: {window_size}, minimum 5 olmalı")
            return
        
        self.window_size = window_size
        logger.info(f"ClusteringModel pencere boyutu {window_size} olarak ayarlandı")