#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tahmin Motoru
-----------
Tüm algoritmaları ve derin öğrenme modelini koordine eden ana tahmin motoru.
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class PredictionEngine:
    """
    Tahmin Motoru
    
    Tüm algoritmaları ve derin öğrenme modelini yöneterek tahmin üretir.
    """
    
    def __init__(self):
        """
        PredictionEngine sınıfını başlatır
        """
        self.algorithms = []  # Kayıtlı algoritmaların listesi
        self.deep_learning_model = None  # LSTM modeli
        self.result_aggregator = None  # Sonuç birleştirici
        self.db_manager = None  # Veritabanı yöneticisi
        
        logger.info("Tahmin motoru başlatıldı")
    
    def register_algorithm(self, algorithm):
        """
        Bir algoritma kaydeder
        
        Args:
            algorithm: Algoritma nesnesi (BaseAlgorithm'dan türetilmiş)
        """
        self.algorithms.append(algorithm)
        logger.info(f"Algoritma kaydedildi: {algorithm.name}")
    
    def set_deep_learning_model(self, model):
        """
        Derin öğrenme modelini ayarlar
        
        Args:
            model: LSTM model nesnesi
        """
        self.deep_learning_model = model
        logger.info("Derin öğrenme modeli ayarlandı")
    
    def set_result_aggregator(self, aggregator):
        """
        Sonuç birleştiriciyi ayarlar
        
        Args:
            aggregator: ResultAggregator nesnesi
        """
        self.result_aggregator = aggregator
        logger.info("Sonuç birleştirici ayarlandı")
    
    def set_db_manager(self, db_manager):
        """
        Veritabanı yöneticisini ayarlar
        
        Args:
            db_manager: DatabaseManager nesnesi
        """
        self.db_manager = db_manager
        logger.info("Veritabanı yöneticisi ayarlandı")
    
    def predict(self, data=None, save_prediction=True):
        """
        Tüm algoritmaları ve derin öğrenme modelini kullanarak tahmin yapar
        
        Args:
            data (dict, optional): Tahmin verileri. None ise veritabanından son veriler çekilir
            save_prediction (bool): Tahmini veritabanına kaydet
        
        Returns:
            dict: Nihai tahmin sonucu
        """
        start_time = time.time()
        
        # Gerekli bileşenleri kontrol et
        if not self.algorithms:
            logger.error("Algoritma bulunamadı. Önce algoritma kaydedin.")
            return None
        
        if not self.result_aggregator:
            logger.error("Sonuç birleştirici bulunamadı. Önce sonuç birleştirici ayarlayın.")
            return None
        
        # Veri yoksa veritabanından son verileri getir
        if data is None:
            if not self.db_manager:
                logger.error("Veritabanı yöneticisi bulunamadı. Veri sağlayın veya DB yöneticisi ayarlayın.")
                return None
            
            last_results = self.db_manager.get_last_n_results(20)
            if not last_results:
                logger.warning("Veritabanında sonuç bulunamadı. Tahmin yapılamıyor.")
                return None
            
            # Sonuçları düzenle
            results = [result['result'] for result in last_results]
            results.reverse()  # En eskiden en yeniye sırala
            
            data = {'last_results': results}
        
        # Her algoritma için tahmin yap
        algorithm_predictions = []
        
        for algorithm in self.algorithms:
            try:
                # Algoritma tahminini al
                pred_result = algorithm.predict(data)
                
                algorithm_predictions.append({
                    'algorithm': algorithm.name,
                    'prediction': pred_result['prediction'],
                    'confidence': pred_result['confidence'],
                    'weight': algorithm.weight
                })
                
                # Güven skorunu güncelle
                algorithm.add_confidence_score(pred_result['confidence'])
                
                logger.debug(f"{algorithm.name} tahmini: {pred_result['prediction']}, Güven: {pred_result['confidence']:.4f}")
            except Exception as e:
                logger.error(f"{algorithm.name} tahmin hatası: {str(e)}")
        
        # Derin öğrenme modelinden tahmin al (eğer mevcutsa ve eğitildiyse)
        if self.deep_learning_model and hasattr(self.deep_learning_model, 'trained') and self.deep_learning_model.trained:
            try:
                # Veriyi model formatına dönüştür
                if 'last_results' in data:
                    # Sonuçları sayısal değerlere dönüştür
                    result_map = {'P': 0, 'B': 1, 'T': 2}
                    numerical_results = [result_map.get(result, 1) for result in data['last_results']]
                    
                    # Model tahminini al
                    dl_prediction = self.deep_learning_model.predict({
                        'sequential': numerical_results,
                        'one_hot': False
                    })
                    
                    # Tahminlere ekle
                    algorithm_predictions.append({
                        'algorithm': 'LSTM',
                        'prediction': dl_prediction['prediction'],
                        'confidence': dl_prediction['confidence'],
                        'weight': 1.5  # Derin öğrenme modeline daha yüksek ağırlık verilir
                    })
                    
                    logger.debug(f"LSTM tahmini: {dl_prediction['prediction']}, Güven: {dl_prediction['confidence']:.4f}")
            except Exception as e:
                logger.error(f"LSTM tahmin hatası: {str(e)}")
        
        # Sonuçları birleştir
        final_prediction = self.result_aggregator.aggregate(algorithm_predictions)
        
        # Her algoritmanın katkısını hesapla
        contributions = self.result_aggregator.get_algorithm_contributions(
            algorithm_predictions, 
            final_prediction['prediction']
        )
        
        # Sonucu zenginleştir
        final_prediction['algorithms'] = algorithm_predictions
        final_prediction['contributions'] = contributions
        final_prediction['timestamp'] = datetime.now().isoformat()
        final_prediction['execution_time'] = time.time() - start_time
        
        # Veritabanına kaydet (opsiyonel)
        if save_prediction and self.db_manager:
            try:
                # Algoritmaları veritabanına kaydet
                for algo_pred in algorithm_predictions:
                    algorithm_name = algo_pred['algorithm']
                    
                    # Algoritma ID'sini al veya oluştur
                    algorithm_info = self.db_manager.get_algorithm_by_name(algorithm_name)
                    if algorithm_info:
                        algorithm_id = algorithm_info['id']
                    else:
                        algorithm_id = self.db_manager.save_algorithm(
                            name=algorithm_name,
                            algorithm_type="traditional" if algorithm_name != "LSTM" else "deep_learning",
                            weight=algo_pred['weight']
                        )
                    
                    # Eğer aktif oturum yoksa yeni bir oturum oluştur
                    # Burada gerçek uygulamada oturum yönetimi yapılabilir
                    session_id = 1
                    
                    # Şu anki verileri kullanarak bir desen oluştur
                    if 'last_results' in data and len(data['last_results']) >= 3:
                        pattern = ''.join(data['last_results'][-3:])
                    else:
                        pattern = ""
                    
                    # Sonuç henüz bilinmediği için tahmin kaydedilir, sonuç daha sonra güncellenebilir
            except Exception as e:
                logger.error(f"Tahmin kaydetme hatası: {str(e)}")
        
        logger.info(f"Nihai tahmin: {final_prediction['prediction']}, Güven: {final_prediction['confidence']:.4f}")
        return final_prediction
    
    def update_results(self, actual_result):
        """
        Gerçek sonucu kaydeder ve algoritma performanslarını günceller
        
        Args:
            actual_result (str): Gerçekleşen sonuç (P/B/T)
        
        Returns:
            int: Eklenen sonuç kaydının ID'si, hata durumunda -1
        """
        if not self.db_manager:
            logger.error("Veritabanı yöneticisi bulunamadı. Sonuç kaydedilemedi.")
            return -1
        
        try:
            # Son tahminleri getir
            # Gerçek uygulamada burada aktif tahminleri almanız gerekebilir
            
            # Sonucu kaydet
            result_id = self.db_manager.save_result(
                result=actual_result,
                timestamp=datetime.now().isoformat(),
                previous_pattern=None,  # Burada önceki deseninizi ekleyebilirsiniz
                session_id=1  # Aktif oturum ID'si
            )
            
            # Algoritma performanslarını güncelle
            # Gerçek uygulamada burada algoritma tahminlerini ve gerçek sonucu karşılaştırmanız gerekir
            
            logger.info(f"Sonuç kaydedildi: ID={result_id}, Sonuç={actual_result}")
            return result_id
        except Exception as e:
            logger.error(f"Sonuç güncelleme hatası: {str(e)}")
            return -1
    
    def get_algorithm_stats(self):
        """
        Tüm algoritmaların istatistiklerini getirir
        
        Returns:
            list: Algoritma istatistiklerinin listesi
        """
        stats = []
        
        for algorithm in self.algorithms:
            stats.append(algorithm.get_info())
        
        # Derin öğrenme modeli istatistiklerini ekle
        if self.deep_learning_model and hasattr(self.deep_learning_model, 'trained'):
            stats.append({
                'name': 'LSTM',
                'weight': 1.5,  # Varsayılan ağırlık
                'accuracy': getattr(self.deep_learning_model, 'accuracy', 0.0),
                'trained': self.deep_learning_model.trained,
                'model_version': getattr(self.deep_learning_model, 'version', '1.0')
            })
        
        return stats