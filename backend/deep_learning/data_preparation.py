#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri Hazırlama Modülü
-------------------
LSTM modeli için eğitim ve test verilerini hazırlar.
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

logger = logging.getLogger(__name__)

class DataPreparation:
    """
    LSTM modeli için veri hazırlama sınıfı
    
    Veritabanından verileri alır, ön işleme yapar ve modele uygun formata dönüştürür.
    """
    
    def __init__(self, db_manager):
        """
        DataPreparation sınıfını başlatır
        
        Args:
            db_manager: Veritabanı yönetici nesnesi
        """
        self.db_manager = db_manager
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        logger.info("Veri hazırlama modülü başlatıldı")
    
    def prepare_sequence_data(self, sequence_length=10, test_size=0.2, include_features=False):
        """
        Sıralı veri setini hazırlar
        
        Args:
            sequence_length (int): Dizi uzunluğu (her bir girdi için kaç önceki sonuç kullanılacak)
            test_size (float): Test verisi oranı (0-1 arası)
            include_features (bool): Ek özellikler eklensin mi
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test) veya None (veri yetersizse)
        """
        try:
            # Tüm sonuçları getir
            all_results = self.db_manager.get_all_results()
            
            if len(all_results) < sequence_length + 1:
                logger.warning(f"Yetersiz veri: {len(all_results)} < {sequence_length + 1}")
                return None
            
            # Sonuçları sayısal değerlere dönüştür (P=0, B=1, T=2)
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = np.array([result_map.get(result, 1) for result in all_results])
            
            # Özellik vektörlerini getir (opsiyonel)
            features = None
            if include_features:
                features = self._extract_additional_features(all_results)
            
            # Giriş-çıkış veri setlerini oluştur
            X, y = [], []
            
            for i in range(len(numerical_results) - sequence_length):
                # Temel dizi
                seq = numerical_results[i:i+sequence_length]
                
                # Ek özellikler ekle
                if include_features and features is not None:
                    seq_features = features[i+sequence_length-1]
                    seq = np.append(seq, seq_features)
                
                X.append(seq)
                y.append(numerical_results[i+sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # One-hot encoding
            y_one_hot = np.zeros((y.size, 3))
            y_one_hot[np.arange(y.size), y] = 1
            
            # Veriyi eğitim ve test setlerine ayır
            X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=test_size, random_state=42)
            
            # Giriş verilerini normalize et
            if not include_features:  # Sadece sıralı verilerde normalize et
                X_train_reshaped = X_train.reshape(-1, 1)
                X_test_reshaped = X_test.reshape(-1, 1)
                
                self.scaler.fit(X_train_reshaped)
                
                X_train_normalized = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
                X_test_normalized = self.scaler.transform(X_test_reshaped).reshape(X_test.shape)
                
                # Veriyi [batch_size, sequence_length, features] formatına dönüştür
                X_train = X_train_normalized.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test_normalized.reshape(X_test.shape[0], X_test.shape[1], 1)
            else:
                # Kompleks özelliklerde farklı bir yaklaşım gerekebilir
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            logger.info(f"Veri hazırlama tamamlandı: {X_train.shape[0]} eğitim örneği, {X_test.shape[0]} test örneği")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {str(e)}")
            return None
    
    def prepare_recent_data(self, sequence_length=10):
        """
        Tahmin için son verileri hazırlar
        
        Args:
            sequence_length (int): Dizi uzunluğu
        
        Returns:
            ndarray: Modele girecek hazırlanmış veri
        """
        try:
            # Son N sonucu getir
            recent_results = self.db_manager.get_last_n_results(sequence_length)
            
            if len(recent_results) < sequence_length:
                logger.warning(f"Yetersiz veri: {len(recent_results)} < {sequence_length}")
                return None
            
            # Sonuçları sayısal değerlere dönüştür
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = [result_map.get(result['result'], 1) for result in recent_results]
            
            # En son sonuçları başa al (kronolojik sıra)
            numerical_results.reverse()
            
            # Veriyi numpy dizisine dönüştür
            X = np.array(numerical_results).reshape(1, sequence_length, 1)
            
            # Eğer scaler eğitildiyse normalize et
            if hasattr(self.scaler, 'data_min_'):
                X_reshaped = X.reshape(-1, 1)
                X_normalized = self.scaler.transform(X_reshaped).reshape(X.shape)
                X = X_normalized
            
            return X
        except Exception as e:
            logger.error(f"Son verileri hazırlama hatası: {str(e)}")
            return None
    
    def _extract_additional_features(self, results):
        """
        Ek özellikler çıkarır (istatistiksel ve desenler)
        
        Args:
            results (list): Sonuç listesi
        
        Returns:
            ndarray: Ek özellikler dizisi
        """
        try:
            features = []
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = [result_map.get(result, 1) for result in results]
            
            for i in range(len(results)):
                # Son birkaç sonuçtaki P, B ve T sayıları
                window_size = 20
                start_idx = max(0, i - window_size)
                window = numerical_results[start_idx:i+1]
                
                p_count = window.count(0) / len(window) if window else 0
                b_count = window.count(1) / len(window) if window else 0
                t_count = window.count(2) / len(window) if window else 0
                
                # Son birkaç sonuçtaki ardışık aynı sonuçlar
                streak = 1
                for j in range(i, max(0, i - 10), -1):
                    if j > 0 and numerical_results[j] == numerical_results[j-1]:
                        streak += 1
                    else:
                        break
                
                normalized_streak = min(streak / 10.0, 1.0)  # 10'dan fazla aynı sonuç nadirdir
                
                # Alternatif sıra (P-B değişimi)
                alternating = 0
                for j in range(i, max(0, i - 10), -1):
                    if j > 0 and numerical_results[j] != numerical_results[j-1]:
                        if numerical_results[j] in [0, 1] and numerical_results[j-1] in [0, 1]:
                            alternating += 1
                
                normalized_alternating = min(alternating / 5.0, 1.0)
                
                # Özellik vektörü
                feature_vector = [p_count, b_count, t_count, normalized_streak, normalized_alternating]
                features.append(feature_vector)
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Özellik çıkarma hatası: {str(e)}")
            return None
    
    def load_and_process_external_data(self, file_path, sequence_length=10):
        """
        Harici CSV veya Excel dosyasından veri yükler ve işler
        
        Args:
            file_path (str): Dosya yolu
            sequence_length (int): Dizi uzunluğu
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        try:
            # Dosya uzantısına göre yükleme yap
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Desteklenmeyen dosya formatı: {file_path}")
                return None
            
            # Sonuç sütununu kontrol et
            if 'result' not in df.columns:
                logger.error("Sonuç sütunu ('result') bulunamadı")
                return None
            
            # Sonuçları işle
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = df['result'].map(lambda x: result_map.get(x, 1)).values
            
            # Dizi veri setini oluştur
            X, y = [], []
            for i in range(len(numerical_results) - sequence_length):
                X.append(numerical_results[i:i+sequence_length])
                y.append(numerical_results[i+sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # One-hot encoding
            y_one_hot = np.zeros((y.size, 3))
            y_one_hot[np.arange(y.size), y] = 1
            
            # Veriyi eğitim ve test setlerine ayır
            X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
            
            # Veriyi [batch_size, sequence_length, features] formatına dönüştür
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            logger.info(f"Harici veri yükleme tamamlandı: {X_train.shape[0]} eğitim örneği, {X_test.shape[0]} test örneği")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Harici veri yükleme hatası: {str(e)}")
            return None