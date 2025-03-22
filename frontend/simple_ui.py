#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baccarat Tahmin Sistemi - Basit Konsol Kullanıcı Arayüzü
-------------------------------------------------------
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Ana dizini ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Backend modüllerini import et
from backend.database.db_manager import DatabaseManager
from backend.engine.prediction_engine import PredictionEngine
from backend.engine.result_aggregator import ResultAggregator
from backend.engine.performance_tracker import PerformanceTracker

# Algoritmaları import et
from backend.algorithms.pattern_analysis import PatternAnalysis
from backend.algorithms.statistical import StatisticalModel
from backend.algorithms.sequence import SequenceAnalysis
from backend.algorithms.bayes import BayesModel
from backend.algorithms.combination import CombinationAnalysis
from backend.algorithms.markov import MarkovModel
from backend.algorithms.cyclical import CyclicalAnalysis
from backend.algorithms.correlation import CorrelationModel
from backend.algorithms.monte_carlo import MonteCarloSimulation
from backend.algorithms.clustering import ClusteringModel
from backend.algorithms.time_series import TimeSeriesModel
from backend.algorithms.entropy import EntropyModel
from backend.algorithms.regression import RegressionModel

# Derin öğrenme modelini import et
from backend.deep_learning.lstm_model import LSTMModel
from backend.deep_learning.training import ModelTrainer

# Konfigürasyon ayarlarını import et
from config.settings import DATABASE_URI, ALGORITHMS, DEEP_LEARNING

class BaccaratConsoleUI:
    """
    Baccarat Tahmin Sistemi için basit konsol kullanıcı arayüzü
    """
    
    def __init__(self):
        """
        BaccaratConsoleUI sınıfını başlatır
        """
        self.db_manager = None
        self.prediction_engine = None
        self.result_aggregator = None
        self.performance_tracker = None
        self.lstm_model = None
        
        self.current_session_id = None
        self.session_results = []
        
        # Renk kodları
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'blue': '\033[94m',
            'yellow': '\033[93m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m',
            'bold': '\033[1m'
        }
    
    def initialize_system(self):
        """
        Sistemi başlatır ve bileşenleri ayarlar
        
        Returns:
            bool: Başlatma başarılı ise True, değilse False
        """
        try:
            print(f"{self.colors['bold']}Baccarat Tahmin Sistemi başlatılıyor...{self.colors['reset']}")
            
            # Veritabanı bağlantısını oluştur
            self.db_manager = DatabaseManager(DATABASE_URI)
            if not self.db_manager.connect():
                print(f"{self.colors['red']}Veritabanı bağlantısı kurulamadı!{self.colors['reset']}")
                return False
            
            # Sonuç birleştiriciyi oluştur
            self.result_aggregator = ResultAggregator()
            
            # Performans izleyiciyi oluştur
            self.performance_tracker = PerformanceTracker(self.db_manager)
            
            # Tahmin motorunu oluştur
            self.prediction_engine = PredictionEngine()
            self.prediction_engine.set_result_aggregator(self.result_aggregator)
            self.prediction_engine.set_db_manager(self.db_manager)
            
            # Tüm algoritmaları kaydettir
            self._register_algorithms()
            
            # LSTM modelini oluştur ve yükle
            if DEEP_LEARNING['enabled']:
                self._initialize_lstm_model()
            
            # Yeni bir oturum oluştur
            self.current_session_id = self.db_manager.create_session()
            
            print(f"{self.colors['green']}Sistem başarıyla başlatıldı!{self.colors['reset']}")
            return True
        except Exception as e:
            print(f"{self.colors['red']}Sistem başlatma hatası: {str(e)}{self.colors['reset']}")
            return False
    
    def _register_algorithms(self):
        """
        Algoritmaları kaydeder
        """
        # PatternAnalysis
        if ALGORITHMS['pattern_analysis']['enabled']:
            self.prediction_engine.register_algorithm(
                PatternAnalysis(
                    weight=ALGORITHMS['pattern_analysis']['weight'],
                    db_manager=self.db_manager,
                    min_samples=ALGORITHMS['pattern_analysis']['min_samples'],
                    pattern_length=ALGORITHMS['pattern_analysis']['pattern_length']
                )
            )
        
        # StatisticalModel
        if ALGORITHMS['statistical_model']['enabled']:
            self.prediction_engine.register_algorithm(
                StatisticalModel(
                    weight=ALGORITHMS['statistical_model']['weight'],
                    db_manager=self.db_manager
                )
            )
        
        # Diğer algoritmaları da benzer şekilde kaydet
        # Gerçek uygulamada tüm algoritmaları ekleyin
        
        print(f"{self.colors['cyan']}Toplam {len(self.prediction_engine.algorithms)} algoritma kaydedildi.{self.colors['reset']}")
    
    def _initialize_lstm_model(self):
        """
        LSTM modelini başlatır
        """
        try:
            self.lstm_model = LSTMModel(
                input_size=DEEP_LEARNING['input_size'],
                hidden_size=DEEP_LEARNING['hidden_size'],
                num_layers=DEEP_LEARNING['num_layers'],
                output_size=DEEP_LEARNING['output_size']
            )
            
            # Daha önce eğitilmiş model varsa yükle
            model_path = os.path.join('models', 'lstm_latest.h5')
            if os.path.exists(model_path):
                self.lstm_model.load_model(model_path)
                print(f"{self.colors['green']}Önceden eğitilmiş LSTM modeli yüklendi.{self.colors['reset']}")
            else:
                print(f"{self.colors['yellow']}Önceden eğitilmiş LSTM modeli bulunamadı, yeni model oluşturuldu.{self.colors['reset']}")
                self.lstm_model.build_model()
            
            # Modeli tahmin motoruna ekle
            self.prediction_engine.set_deep_learning_model(self.lstm_model)
            
            # Modeli yeniden eğitme kontrolü
            last_n_records = self.db_manager.get_last_n_results(1000)
            if len(last_n_records) >= 200:  # En az 200 kayıt varsa eğitime başla
                print(f"{self.colors['yellow']}Yeterli veri bulundu, LSTM modeli eğitiliyor...{self.colors['reset']}")
                model_trainer = ModelTrainer(self.lstm_model, self.db_manager)
                model_trainer.train_model(epochs=DEEP_LEARNING['epochs'], batch_size=DEEP_LEARNING['batch_size'])
        except Exception as e:
            print(f"{self.colors['red']}LSTM modeli başlatma hatası: {str(e)}{self.colors['reset']}")
    
    def display_menu(self):
        """
        Ana menüyü görüntüler
        """
        print(f"\n{self.colors['bold']}===== BACCARAT TAHMİN SİSTEMİ ====={self.colors['reset']}")
        print(f"1. {self.colors['blue']}Player{self.colors['reset']} sonucu gir")
        print(f"2. {self.colors['red']}Banker{self.colors['reset']} sonucu gir")
        print(f"3. {self.colors['green']}Tie{self.colors['reset']} sonucu gir")
        print(f"4. Mevcut tahminleri görüntüle")
        print(f"5. Son oyun geçmişini görüntüle")
        print(f"6. Algoritma performanslarını görüntüle")
        print(f"7. Toplu sonuç girişi")
        print(f"8. Derin öğrenme modelini yeniden eğit")
        print(f"9. Algoritma ağırlıklarını optimize et")
        print(f"0. Çıkış")
        print(f"{self.colors['bold']}=================================={self.colors['reset']}")
    
    def run(self):
        """
        Kullanıcı arayüzünü çalıştırır
        """
        if not self.initialize_system():
            print(f"{self.colors['red']}Sistem başlatılamadı, çıkılıyor...{self.colors['reset']}")
            return
        
        # Ana program döngüsü
        while True:
            self.display_menu()
            choice = input("Seçiminiz: ")
            
            if choice == '1':
                self.enter_result('P')
            elif choice == '2':
                self.enter_result('B')
            elif choice == '3':
                self.enter_result('T')
            elif choice == '4':
                self.show_current_prediction()
            elif choice == '5':
                self.show_game_history()
            elif choice == '6':
                self.show_algorithm_performance()
            elif choice == '7':
                self.enter_bulk_results()
            elif choice == '8':
                self.retrain_lstm_model()
            elif choice == '9':
                self.optimize_weights()
            elif choice == '0':
                self.exit_program()
                break
            else:
                print(f"{self.colors['red']}Geçersiz seçenek, lütfen tekrar deneyin.{self.colors['reset']}")
    
    def enter_result(self, result):
        """
        Yeni bir oyun sonucu girer
        
        Args:
            result (str): Oyun sonucu (P/B/T)
        """
        # Önce mevcut tahminleri göster
        prediction = self.show_current_prediction()
        
        if prediction:
            # Sonucu kaydet
            result_id = self.prediction_engine.update_results(result)
            
            if result_id > 0:
                # Sonuç başarıyla kaydedildi
                result_color = self.get_result_color(result)
                print(f"\n{result_color}Sonuç kaydedildi: {result}{self.colors['reset']}")
                
                # Tahmin doğru muydu?
                is_correct = prediction['prediction'] == result
                correct_str = f"{self.colors['green']}DOĞRU{self.colors['reset']}" if is_correct else f"{self.colors['red']}YANLIŞ{self.colors['reset']}"
                print(f"Tahmin: {prediction['prediction']} - {correct_str}")
                
                # Oturum sonuçlarına ekle
                self.session_results.append({
                    'id': result_id,
                    'result': result,
                    'prediction': prediction['prediction'],
                    'is_correct': is_correct
                })
                
                # İstatistikleri güncelle
                correct_count = sum(1 for r in self.session_results if r['is_correct'])
                total_count = len(self.session_results)
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                print(f"\nOturum İstatistikleri:")
                print(f"Toplam: {total_count}, Doğru: {correct_count}, Doğruluk: {accuracy:.2f}")
                
                # Bir sonraki tahmini hesapla
                print(f"\n{self.colors['yellow']}Bir sonraki tahmin hesaplanıyor...{self.colors['reset']}")
                time.sleep(1)  # Kısa bir bekleme
                self.show_current_prediction()
            else:
                print(f"{self.colors['red']}Sonuç kaydedilemedi!{self.colors['reset']}")
    
    def show_current_prediction(self):
        """
        Mevcut tahmini görüntüler
        
        Returns:
            dict: Tahmin sonucu
        """
        print(f"\n{self.colors['bold']}Tahmin hesaplanıyor...{self.colors['reset']}")
        
        # Tahmin yap
        prediction = self.prediction_engine.predict(save_prediction=False)
        
        if not prediction:
            print(f"{self.colors['red']}Tahmin yapılamadı! Yeterli veri yok.{self.colors['reset']}")
            return None
        
        # Tahmin sonucunu görüntüle
        predicted_result = prediction['prediction']
        confidence = prediction['confidence']
        details = prediction['details']
        
        result_color = self.get_result_color(predicted_result)
        print(f"\n{self.colors['bold']}MEVCUT TAHMİN:{self.colors['reset']}")
        print(f"{result_color}Tahmin: {predicted_result}{self.colors['reset']} - Güven: {confidence:.2f}")
        
        # Detaylı olasılıkları göster
        print(f"\nDetaylı Olasılıklar:")
        print(f"{self.colors['blue']}Player (P): {details.get('P', 0):.2f}{self.colors['reset']}")
        print(f"{self.colors['red']}Banker (B): {details.get('B', 0):.2f}{self.colors['reset']}")
        print(f"{self.colors['green']}Tie    (T): {details.get('T', 0):.2f}{self.colors['reset']}")
        
        # Algoritma katkılarını göster
        print(f"\nAlgoritma Katkıları:")
        for algo in prediction['algorithms']:
            algo_name = algo['algorithm']
            algo_pred = algo['prediction']
            algo_conf = algo['confidence']
            
            # Katkı rengini belirle (aynı tahmin için yeşil, farklı için kırmızı)
            color = self.colors['green'] if algo_pred == predicted_result else self.colors['red']
            print(f"{color}{algo_name}: {algo_pred} ({algo_conf:.2f}){self.colors['reset']}")
        
        return prediction
    
    def show_game_history(self, count=20):
        """
        Oyun geçmişini görüntüler
        
        Args:
            count (int): Görüntülenecek sonuç sayısı
        """
        print(f"\n{self.colors['bold']}SON {count} OYUN SONUCU:{self.colors['reset']}")
        
        # Son n sonucu getir
        results = self.db_manager.get_last_n_results(count)
        
        if not results:
            print(f"{self.colors['yellow']}Henüz kayıtlı sonuç yok.{self.colors['reset']}")
            return
        
        # Sonuçları görüntüle
        print(f"{'ID':<5} {'TARİH':<20} {'SONUÇ':<10}")
        print("-" * 40)
        
        for result in results:
            result_id = result['id']
            timestamp = result['timestamp']
            result_value = result['result']
            
            result_color = self.get_result_color(result_value)
            print(f"{result_id:<5} {timestamp:<20} {result_color}{result_value}{self.colors['reset']}")
        
        # Özet istatistikler
        p_count = sum(1 for r in results if r['result'] == 'P')
        b_count = sum(1 for r in results if r['result'] == 'B')
        t_count = sum(1 for r in results if r['result'] == 'T')
        
        print(f"\nÖzet:")
        print(f"{self.colors['blue']}Player (P): {p_count} ({p_count/len(results)*100:.1f}%){self.colors['reset']}")
        print(f"{self.colors['red']}Banker (B): {b_count} ({b_count/len(results)*100:.1f}%){self.colors['reset']}")
        print(f"{self.colors['green']}Tie    (T): {t_count} ({t_count/len(results)*100:.1f}%){self.colors['reset']}")
    
    def show_algorithm_performance(self):
        """
        Algoritma performanslarını görüntüler
        """
        print(f"\n{self.colors['bold']}ALGORİTMA PERFORMANSLARI:{self.colors['reset']}")
        
        # Algoritma istatistiklerini al
        stats = self.prediction_engine.get_algorithm_stats()
        
        if not stats:
            print(f"{self.colors['yellow']}Performans verisi bulunamadı.{self.colors['reset']}")
            return
        
        # Performansları görüntüle
        print(f"{'ALGORİTMA':<25} {'DOĞRULUK':<10} {'AĞIRLIK':<10}")
        print("-" * 50)
        
        for stat in stats:
            name = stat['name']
            accuracy = stat.get('accuracy', 0.0)
            weight = stat.get('weight', 1.0)
            
            # Doğruluk oranına göre renk belirle
            if accuracy > 0.7:
                acc_color = self.colors['green']
            elif accuracy > 0.5:
                acc_color = self.colors['yellow']
            else:
                acc_color = self.colors['red']
            
            print(f"{name:<25} {acc_color}{accuracy:.4f}{self.colors['reset']} {weight:.2f}")
    
    def enter_bulk_results(self):
        """
        Toplu sonuç girişi yapar
        """
        print(f"\n{self.colors['bold']}TOPLU SONUÇ GİRİŞİ{self.colors['reset']}")
        print(f"P: Player, B: Banker, T: Tie olacak şekilde sonuçları girin.")
        print(f"Örnek: PPBBTPBPBP")
        
        results = input("Sonuçlar: ").upper()
        
        valid_results = []
        for char in results:
            if char in ['P', 'B', 'T']:
                valid_results.append(char)
        
        if not valid_results:
            print(f"{self.colors['red']}Geçerli sonuç bulunamadı. Lütfen P, B, T harflerini kullanın.{self.colors['reset']}")
            return
        
        # Sonuçları kaydet
        print(f"{self.colors['yellow']}Toplam {len(valid_results)} sonuç kaydediliyor...{self.colors['reset']}")
        
        for result in valid_results:
            result_id = self.prediction_engine.update_results(result)
            if result_id > 0:
                print(f"{self.get_result_color(result)}{result}{self.colors['reset']} sonucu kaydedildi.")
            else:
                print(f"{self.colors['red']}{result} sonucu kaydedilemedi!{self.colors['reset']}")
        
        print(f"{self.colors['green']}Toplu sonuç girişi tamamlandı.{self.colors['reset']}")
    
    def retrain_lstm_model(self):
        """
        LSTM modelini yeniden eğitir
        """
        if not DEEP_LEARNING['enabled'] or not self.lstm_model:
            print(f"{self.colors['red']}LSTM modeli etkin değil!{self.colors['reset']}")
            return
        
        print(f"\n{self.colors['bold']}LSTM MODELİ YENİDEN EĞİTİLİYOR...{self.colors['reset']}")
        
        # Model eğitimi için yeterli veri var mı kontrol et
        last_results = self.db_manager.get_last_n_results(1000)
        if len(last_results) < 200:
            print(f"{self.colors['red']}Yeterli veri yok! En az 200 sonuç gerekli.{self.colors['reset']}")
            return
        
        # Modeli yeniden eğit
        model_trainer = ModelTrainer(self.lstm_model, self.db_manager)
        history = model_trainer.train_model(
            epochs=DEEP_LEARNING['epochs'],
            batch_size=DEEP_LEARNING['batch_size'],
            force=True
        )
        
        if history:
            print(f"{self.colors['green']}Model başarıyla yeniden eğitildi!{self.colors['reset']}")
        else:
            print(f"{self.colors['red']}Model eğitimi başarısız oldu!{self.colors['reset']}")
    
    def optimize_weights(self):
        """
        Algoritma ağırlıklarını optimize eder
        """
        print(f"\n{self.colors['bold']}ALGORİTMA AĞIRLIKLARI OPTİMİZE EDİLİYOR...{self.colors['reset']}")
        
        # Ağırlıkları optimize et
        updated_weights = self.performance_tracker.optimize_weights()
        
        if not updated_weights:
            print(f"{self.colors['red']}Ağırlık optimizasyonu başarısız oldu!{self.colors['reset']}")
            return
        
        # Güncellenmiş ağırlıkları görüntüle
        print(f"\n{'ALGORİTMA':<25} {'ESKİ AĞIRLIK':<15} {'YENİ AĞIRLIK':<15}")
        print("-" * 60)
        
        for algo_name, data in updated_weights.items():
            old_weight = data['old_weight']
            new_weight = data['new_weight']
            
            # Ağırlık değişimine göre renk belirle
            if new_weight > old_weight:
                weight_color = self.colors['green']
            elif new_weight < old_weight:
                weight_color = self.colors['red']
            else:
                weight_color = self.colors['reset']
            
            print(f"{algo_name:<25} {old_weight:<15.2f} {weight_color}{new_weight:<15.2f}{self.colors['reset']}")
        
        print(f"\n{self.colors['green']}Ağırlıklar başarıyla optimize edildi!{self.colors['reset']}")
    
    def exit_program(self):
        """
        Programdan çıkar
        """
        # Oturumu sonlandır
        if self.current_session_id:
            correct_count = sum(1 for r in self.session_results if r['is_correct'])
            total_count = len(self.session_results)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            self.db_manager.end_session(
                session_id=self.current_session_id,
                total_games=total_count,
                win_rate=accuracy
            )
        
        print(f"\n{self.colors['bold']}Oturum sonlandırılıyor...{self.colors['reset']}")
        
        # Veritabanı bağlantısını kapat
        if self.db_manager:
            self.db_manager.disconnect()
        
        print(f"{self.colors['green']}Program sonlandırıldı. İyi günler!{self.colors['reset']}")
    
    def get_result_color(self, result):
        """
        Sonuç değerine göre renk kodu döndürür
        
        Args:
            result (str): Sonuç değeri (P/B/T)
        
        Returns:
            str: Renk kodu
        """
        if result == 'P':
            return self.colors['blue']
        elif result == 'B':
            return self.colors['red']
        elif result == 'T':
            return self.colors['green']
        return self.colors['reset']

def main():
    """
    Ana program başlangıç noktası
    """
    parser = argparse.ArgumentParser(description='Baccarat Tahmin Sistemi Konsol Arayüzü')
    parser.add_argument('--reset-db', action='store_true', help='Veritabanını sıfırla')
    args = parser.parse_args()
    
    # Veritabanını sıfırla
    if args.reset_db:
        db_manager = DatabaseManager(DATABASE_URI)
        db_manager.reset_database()
        print("Veritabanı sıfırlandı.")
        return
    
    # Kullanıcı arayüzünü başlat
    ui = BaccaratConsoleUI()
    ui.run()

if __name__ == "__main__":
    main()