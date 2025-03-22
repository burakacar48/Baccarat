#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Temel Algoritma Sınıfı
----------------------
Tüm tahmin algoritmalarının temel sınıfı
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAlgorithm(ABC):
    """
    Tüm tahmin algoritmaları için temel sınıf
    
    Her tahmin algoritması bu sınıftan türetilmeli ve
    gerekli metodları (özellikle predict) uygulamalıdır.
    """
    
    def __init__(self, name, weight=1.0):
        """
        BaseAlgorithm sınıfını başlatır
        
        Args:
            name (str): Algoritmanın adı
            weight (float): Algoritmanın ağırlığı (0-10 arası)
        """
        self.name = name
        self.weight = weight
        self.accuracy = 0.0  # Algoritmanın doğruluk oranı
        self.total_predictions = 0  # Toplam tahmin sayısı
        self.correct_predictions = 0  # Doğru tahmin sayısı
        self.last_update = datetime.now()  # Son güncelleme zamanı
        self.confidence_history = []  # Güven skorları geçmişi
        
        logger.info(f"Algoritma Başlatıldı: {name}, Ağırlık: {weight}")
    
    @abstractmethod
    def predict(self, data):
        """
        Verilen verilere göre tahmin yapar
        
        Args:
            data (dict): Tahmin için gerekli veriler
        
        Returns:
            dict: Tahmin sonucu ve ek bilgiler
        """
        pass
    
    @abstractmethod
    def get_confidence(self, data):
        """
        Tahminin güven skorunu hesaplar
        
        Args:
            data (dict): Tahmin için gerekli veriler
        
        Returns:
            float: Güven skoru (0-1 arası)
        """
        pass
    
    def update_metrics(self, prediction, actual_result):
        """
        Algoritma performans metriklerini günceller
        
        Args:
            prediction (str): Algoritmanın tahmini
            actual_result (str): Gerçekleşen sonuç
        """
        self.total_predictions += 1
        
        if prediction == actual_result:
            self.correct_predictions += 1
            logger.debug(f"{self.name}: Doğru tahmin - {prediction}")
        else:
            logger.debug(f"{self.name}: Yanlış tahmin - Tahmin: {prediction}, Gerçek: {actual_result}")
        
        # Doğruluk oranını güncelle
        self.accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        
        # Son güncelleme zamanını güncelle
        self.last_update = datetime.now()
        
        logger.info(f"{self.name} metrikleri güncellendi - Doğruluk: {self.accuracy:.4f}, Toplam: {self.total_predictions}")
    
    def set_weight(self, weight):
        """
        Algoritmanın ağırlığını günceller
        
        Args:
            weight (float): Yeni ağırlık değeri
        """
        # Ağırlık değeri kontrolü
        if weight < 0:
            logger.warning(f"{self.name}: Geçersiz ağırlık değeri: {weight}, minimum 0 olarak ayarlanıyor")
            weight = 0
        
        self.weight = weight
        logger.info(f"{self.name}: Ağırlık {weight} olarak güncellendi")
    
    def add_confidence_score(self, confidence):
        """
        Güven skoru geçmişine yeni bir değer ekler
        
        Args:
            confidence (float): Güven skoru
        """
        self.confidence_history.append(confidence)
        
        # Sadece son 100 skoru sakla
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
    
    def get_average_confidence(self):
        """
        Ortalama güven skorunu hesaplar
        
        Returns:
            float: Ortalama güven skoru
        """
        if not self.confidence_history:
            return 0.0
        
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def get_weighted_score(self, confidence):
        """
        Ağırlıklı skor hesaplar (ağırlık * güven skoru * doğruluk)
        
        Args:
            confidence (float): Güven skoru
        
        Returns:
            float: Ağırlıklı skor
        """
        # Ağırlık, güven skoru ve doğruluk oranının çarpımı
        return self.weight * confidence * self.accuracy if self.total_predictions > 10 else self.weight * confidence * 0.5
    
    def get_info(self):
        """
        Algoritma hakkında bilgi verir
        
        Returns:
            dict: Algoritma bilgileri
        """
        return {
            'name': self.name,
            'weight': self.weight,
            'accuracy': self.accuracy,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'average_confidence': self.get_average_confidence(),
            'last_update': self.last_update.isoformat()
        }