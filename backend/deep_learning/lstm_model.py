#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM Derin Öğrenme Modeli
-------------------------
Baccarat tahminleri için LSTM (Long Short-Term Memory) modeli.
"""

import os
import logging
import numpy as np
import json
from datetime import datetime

# TensorFlow ve Keras kütüphanelerini import et
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow/Keras kütüphanesi bulunamadı. LSTM modeli çalışmayacak.")

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    Baccarat tahminleri için LSTM derin öğrenme modeli
    
    Bu model, geçmiş oyun sonuçlarını kullanarak gelecek sonuçları tahmin eder.
    """
    
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, output_size=3):
        """
        LSTMModel sınıfını başlatır
        
        Args:
            input_size (int): Giriş boyutu (son kaç oyun sonucunun kullanılacağı)
            hidden_size (int): Gizli katman boyutu
            num_layers (int): LSTM katman sayısı
            output_size (int): Çıkış boyutu (P/B/T için 3)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.model = None
        self.model_path = None
        self.version = "1.0"
        self.trained = False
        self.training_history = None
        
        # TensorFlow kullanılabilir değilse uyarı ver
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow/Keras kullanılamıyor. Model oluşturulamayacak.")
            return
        
        # Seed değerini sabitleyerek sonuçları tekrarlanabilir yap
        np.random.seed(42)
        tf.random.set_seed(42)
        
        logger.info(f"LSTM modeli başlatıldı: Giriş={input_size}, Gizli={hidden_size}, Katman={num_layers}, Çıkış={output_size}")
    
    def build_model(self):
        """
        LSTM modelini oluşturur
        
        Returns:
            bool: Model başarıyla oluşturulduysa True, değilse False
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow/Keras kullanılamıyor. Model oluşturulamıyor.")
            return False
        
        try:
            # Sıralı model oluştur
            self.model = Sequential()
            
            # İlk LSTM katmanı
            self.model.add(LSTM(
                units=self.hidden_size,
                input_shape=(self.input_size, 1),
                return_sequences=self.num_layers > 1
            ))
            self.model.add(Dropout(0.2))
            
            # Ara LSTM katmanları
            for i in range(1, self.num_layers - 1):
                self.model.add(LSTM(
                    units=self.hidden_size,
                    return_sequences=True
                ))
                self.model.add(Dropout(0.2))
            
            # Son LSTM katmanı (eğer birden fazla katman varsa)
            if self.num_layers > 1:
                self.model.add(LSTM(units=self.hidden_size))
                self.model.add(Dropout(0.2))
            
            # Çıkış katmanı (softmax ile P/B/T olasılıkları)
            self.model.add(Dense(units=self.output_size, activation='softmax'))
            
            # Modeli derle
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Model özetini logla
            self.model.summary(print_fn=lambda x: logger.debug(x))
            
            logger.info("LSTM modeli başarıyla oluşturuldu")
            return True
        except Exception as e:
            logger.error(f"LSTM modeli oluşturma hatası: {str(e)}")
            return False
    
    def train(self, train_data, val_data=None, epochs=50, batch_size=32):
        """
        Modeli eğitir
        
        Args:
            train_data (tuple): (X_train, y_train) eğitim verileri
            val_data (tuple, optional): (X_val, y_val) doğrulama verileri
            epochs (int): Eğitim döngüsü sayısı
            batch_size (int): Batch boyutu
        
        Returns:
            dict: Eğitim geçmişi
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("Model eğitilemez. TensorFlow/Keras kullanılamıyor veya model oluşturulmadı.")
            return None
        
        try:
            X_train, y_train = train_data
            
            # Modelin kaydetmek için klasör oluştur
            model_dir = os.path.join(os.getcwd(), 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Model kaydetme için dosya yolu oluştur
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.model_path = os.path.join(model_dir, f"lstm_model_{timestamp}.h5")
            
            # Callbacks oluştur
            callbacks = [
                # Erken durdurma (validation loss iyileşmezse)
                EarlyStopping(
                    monitor='val_loss' if val_data else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                # En iyi modeli kaydet
                ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_accuracy' if val_data else 'accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
            
            # Modeli eğit
            if val_data:
                X_val, y_val = val_data
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=2
                )
            else:
                # Eğitim verilerinin %20'sini doğrulama verisi olarak kullan
                history = self.model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=2
                )
            
            # Eğitim geçmişini kaydet
            self.training_history = {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
            
            # Eğitimi tamamla
            self.trained = True
            
            # Son epoch doğruluk değerini logla
            final_accuracy = history.history['val_accuracy'][-1] if val_data else history.history['accuracy'][-1]
            logger.info(f"LSTM modeli eğitimi tamamlandı: Doğruluk={final_accuracy:.4f}, Epoch={len(history.history['accuracy'])}")
            
            return self.training_history
        except Exception as e:
            logger.error(f"LSTM modeli eğitim hatası: {str(e)}")
            return None
    
    def predict(self, input_data):
        """
        Tahmin yapar
        
        Args:
            input_data: Tahmin için giriş verileri
                - sequential (np.array): Son N oyun sonucu dizisi
                - one_hot (bool): Sonuçların one-hot kodlanmış olup olmadığı
        
        Returns:
            dict: Tahmin sonucu
                - prediction (str): Tahmin edilen sonuç (P/B/T)
                - probabilities (dict): Her sınıf için olasılıklar
                - confidence (float): Güven skoru (en yüksek olasılık)
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("Tahmin yapılamaz. TensorFlow/Keras kullanılamıyor veya model oluşturulmadı.")
            return {
                'prediction': 'B',
                'probabilities': {'P': 0.4932, 'B': 0.5068, 'T': 0.0},
                'confidence': 0.5068
            }
    
    def save_model(self, path=None):
        """
        Modeli kaydeder
        
        Args:
            path (str, optional): Kaydedilecek dosya yolu. None ise self.model_path kullanılır
        
        Returns:
            bool: Kaydetme başarılı ise True, değilse False
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("Model kaydedilemez. TensorFlow/Keras kullanılamıyor veya model oluşturulmadı.")
            return False
        
        try:
            # Eğer yol belirtilmemişse mevcut yolu kullan veya oluştur
            if path is None:
                if self.model_path is None:
                    model_dir = os.path.join(os.getcwd(), 'models')
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.model_path = os.path.join(model_dir, f"lstm_model_{timestamp}.h5")
                
                path = self.model_path
            
            # Modeli kaydet
            self.model.save(path)
            logger.info(f"LSTM modeli kaydedildi: {path}")
            
            # Meta verileri kaydet
            meta_path = path.replace('.h5', '_meta.json')
            meta_data = {
                'version': self.version,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_size': self.output_size,
                'trained': self.trained,
                'training_history': self.training_history
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            logger.debug(f"LSTM model meta verileri kaydedildi: {meta_path}")
            return True
        except Exception as e:
            logger.error(f"LSTM modeli kaydetme hatası: {str(e)}")
            return False
    
    def load_model(self, path):
        """
        Modeli yükler
        
        Args:
            path (str): Yüklenecek model dosya yolu
        
        Returns:
            bool: Yükleme başarılı ise True, değilse False
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("Model yüklenemez. TensorFlow/Keras kullanılamıyor.")
            return False
        
        try:
            # Modeli yükle
            self.model = load_model(path)
            self.model_path = path
            
            # Meta verileri yükle
            meta_path = path.replace('.h5', '_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                
                self.version = meta_data.get('version', self.version)
                self.input_size = meta_data.get('input_size', self.input_size)
                self.hidden_size = meta_data.get('hidden_size', self.hidden_size)
                self.num_layers = meta_data.get('num_layers', self.num_layers)
                self.output_size = meta_data.get('output_size', self.output_size)
                self.trained = meta_data.get('trained', True)
                self.training_history = meta_data.get('training_history', None)
            else:
                self.trained = True
            
            logger.info(f"LSTM modeli yüklendi: {path}")
            return True
        except Exception as e:
            logger.error(f"LSTM modeli yükleme hatası: {str(e)}")
            return False
    
    def evaluate(self, test_data):
        """
        Modeli değerlendirir
        
        Args:
            test_data (tuple): (X_test, y_test) test verileri
        
        Returns:
            dict: Değerlendirme sonuçları
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("Model değerlendirilemez. TensorFlow/Keras kullanılamıyor veya model oluşturulmadı.")
            return None
        
        try:
            X_test, y_test = test_data
            
            # Modeli değerlendir
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Sonuçları logla
            logger.info(f"LSTM modeli değerlendirme: Kayıp={loss:.4f}, Doğruluk={accuracy:.4f}")
            
            # Tahminleri al
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Sınıflandırma raporu
            class_indices = {0: 'P', 1: 'B', 2: 'T'}
            
            # Karmaşıklık matrisi
            confusion_matrix = np.zeros((3, 3), dtype=int)
            for true_class, pred_class in zip(y_true_classes, y_pred_classes):
                confusion_matrix[true_class, pred_class] += 1
            
            # Her sınıf için metrikler
            class_metrics = {}
            for i in range(3):
                tp = confusion_matrix[i, i]
                fp = np.sum(confusion_matrix[:, i]) - tp
                fn = np.sum(confusion_matrix[i, :]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[class_indices[i]] = {
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1
                }
            
            return {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'class_metrics': class_metrics,
                'confusion_matrix': confusion_matrix.tolist()
            }
        except Exception as e:
            logger.error(f"LSTM modeli değerlendirme hatası: {str(e)}")
            return None
        
        try:
            # Veri boyutunu kontrol et ve yeniden şekillendir
            if isinstance(input_data, dict):
                if 'sequential' in input_data:
                    data = input_data['sequential']
                    one_hot = input_data.get('one_hot', False)
                else:
                    logger.error("Tahmin giriş verisi 'sequential' anahtarı içermiyor")
                    return {
                        'prediction': 'B',
                        'probabilities': {'P': 0.4932, 'B': 0.5068, 'T': 0.0},
                        'confidence': 0.5068
                    }
            else:
                data = input_data
                one_hot = False
            
            # One-hot kodlamasını kontrol et
            if one_hot:
                # Eğer veriler one-hot kodlanmışsa, indekslere dönüştür
                data = np.argmax(data, axis=1)
            
            # Veriyi modelin beklediği formata dönüştür
            if len(data) < self.input_size:
                # Veri yeterli uzunlukta değilse, başa sıfırlar ekle
                padding = np.zeros(self.input_size - len(data))
                data = np.concatenate([padding, data])
            elif len(data) > self.input_size:
                # Veri çok uzunsa, son input_size kadarını al
                data = data[-self.input_size:]
            
            # Veriyi [batch_size, sequence_length, features] boyutuna yeniden şekillendir
            data = np.array(data).reshape(1, self.input_size, 1)
            
            # Tahmin yap
            prob_prediction = self.model.predict(data)[0]
            
            # Sonuçları yorumla
            class_indices = {0: 'P', 1: 'B', 2: 'T'}
            probabilities = {class_indices[i]: float(prob) for i, prob in enumerate(prob_prediction)}
            
            # En yüksek olasılığa sahip sınıfı belirle
            predicted_class = class_indices[np.argmax(prob_prediction)]
            confidence = float(max(prob_prediction))
            
            logger.debug(f"LSTM tahmin: {predicted_class}, Güven: {confidence:.4f}")
            return {
                'prediction': predicted_class,
                'probabilities': probabilities,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"LSTM tahmin hatası: {str(e)}")
            return {
                'prediction': 'B',
                'probabilities': {'P': 0.4932, 'B': 0.5068, 'T': 0.0},
                'confidence': 0.5068
            }