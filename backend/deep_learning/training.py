#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM Model Eğitim Modülü
------------------------
LSTM derin öğrenme modelinin eğitim sürecini yönetir.
"""

import os
import logging
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    LSTM modelinin eğitim sürecini yöneten sınıf
    
    Veritabanından verileri çeker, ön işleme yapar ve modeli eğitir.
    """
    
    def __init__(self, lstm_model, db_manager):
        """
        ModelTrainer sınıfını başlatır
        
        Args:
            lstm_model: LSTM model nesnesi
            db_manager: Veritabanı yönetici nesnesi
        """
        self.lstm_model = lstm_model
        self.db_manager = db_manager
        logger.info("Model eğitim modülü başlatıldı")
    
    def prepare_training_data(self, sequence_length=10, test_size=0.2):
        """
        Eğitim verilerini hazırlar
        
        Args:
            sequence_length (int): Dizi uzunluğu (giriş vektörü boyutu)
            test_size (float): Test verisi oranı
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        try:
            logger.info("Eğitim verileri hazırlanıyor...")
            
            # Tüm sonuçları getir
            all_results = self.db_manager.get_all_results()
            
            if len(all_results) < sequence_length + 1:
                logger.warning(f"Yetersiz veri: {len(all_results)} < {sequence_length + 1}")
                return None
            
            # Sonuçları sayısal değerlere dönüştür (P=0, B=1, T=2)
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = [result_map.get(result, 1) for result in all_results]  # Bilinmeyen değerler için B (1)
            
            X, y = [], []
            
            # Eğitim veri setini oluştur
            for i in range(len(numerical_results) - sequence_length):
                X.append(numerical_results[i:i + sequence_length])
                y.append(numerical_results[i + sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # One-hot encoding
            y_one_hot = np.zeros((y.size, 3))
            y_one_hot[np.arange(y.size), y] = 1
            
            # Veriyi eğitim ve test setlerine ayır
            X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=test_size, random_state=42)
            
            # Veriyi [batch_size, sequence_length, features] formatına dönüştür
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            logger.info(f"Veri hazırlama tamamlandı: {X_train.shape[0]} eğitim örneği, {X_test.shape[0]} test örneği")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {str(e)}")
            return None
    
    def train_model(self, epochs=50, batch_size=32, force=False):
        """
        LSTM modelini eğitir
        
        Args:
            epochs (int): Eğitim döngüsü sayısı
            batch_size (int): Batch boyutu
            force (bool): True ise, model daha önce eğitilmiş olsa bile yeniden eğitir
        
        Returns:
            dict: Eğitim geçmişi
        """
        # Model daha önce eğitilmiş ve force=False ise eğitimi atla
        if self.lstm_model.trained and not force:
            logger.info("Model zaten eğitilmiş. Yeniden eğitmek için force=True kullanın.")
            return None
        
        # Veri setini hazırla
        data = self.prepare_training_data(sequence_length=self.lstm_model.input_size)
        
        if data is None:
            logger.error("Eğitim verileri hazırlanamadı.")
            return None
        
        X_train, y_train, X_test, y_test = data
        
        # Model oluşturulmamışsa yeni bir model oluştur
        if self.lstm_model.model is None:
            if not self.lstm_model.build_model():
                logger.error("Model oluşturulamadı.")
                return None
        
        # Modeli eğit
        logger.info(f"LSTM modeli eğitiliyor: {epochs} döngü, {batch_size} batch boyutu")
        history = self.lstm_model.train(
            train_data=(X_train, y_train),
            val_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )
        
        if history is None:
            logger.error("Model eğitimi başarısız oldu.")
            return None
        
        # Modeli değerlendir
        evaluation = self.lstm_model.evaluate((X_test, y_test))
        
        if evaluation:
            accuracy = evaluation['accuracy']
            logger.info(f"Model değerlendirme: Doğruluk={accuracy:.4f}")
            
            # Modeli veritabanına kaydet
            model_path = self.lstm_model.model_path or os.path.join('models', 'lstm_latest.h5')
            self.lstm_model.save_model(model_path)
            
            # Model versiyonunu veritabanına kaydet
            self.db_manager.save_model_version(
                model_type="LSTM",
                file_path=model_path,
                accuracy=accuracy,
                is_active=True
            )
        
        return history
    
    def generate_features(self, game_result_id, sequence_length=None):
        """
        Belirli bir oyun sonucu için özellik vektörü oluşturur
        
        Args:
            game_result_id (int): Oyun sonucu ID'si
            sequence_length (int, optional): Dizi uzunluğu, None ise model.input_size kullanılır
        
        Returns:
            dict: Özellik vektörü
        """
        if sequence_length is None:
            sequence_length = self.lstm_model.input_size
        
        try:
            # Son N oyun sonucunu getir
            game_results = self.db_manager.get_last_n_results(sequence_length)
            
            if len(game_results) < sequence_length:
                logger.warning(f"Yetersiz sonuç: {len(game_results)} < {sequence_length}")
                return None
            
            # Sonuçları sayısal değerlere dönüştür
            result_map = {'P': 0, 'B': 1, 'T': 2}
            numerical_results = [result_map.get(result['result'], 1) for result in game_results]
            
            # Ters çevir (en eski -> en yeni)
            numerical_results.reverse()
            
            # Özellik vektörünü oluştur
            features = {
                'sequence': numerical_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Özellik vektörünü veritabanına kaydet
            model_version = self.db_manager.get_active_model("LSTM")
            model_version_id = model_version['id'] if model_version else None
            
            self.db_manager.save_deep_learning_features(
                game_result_id=game_result_id,
                feature_vector=features,
                model_version_id=model_version_id
            )
            
            return features
        except Exception as e:
            logger.error(f"Özellik vektörü oluşturma hatası: {str(e)}")
            return None