#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baccarat Tahmin Sistemi - PyQt5 Grafik Kullanıcı Arayüzü
-------------------------------------------------------
"""

import os
import sys
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
                           QTabWidget, QGroupBox, QRadioButton, QProgressBar, QHeaderView,
                           QMessageBox, QSplitter, QFrame, QComboBox, QDialog, QFormLayout,
                           QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, 
                           QSizePolicy, QGridLayout, QScrollArea, QFileDialog)
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette, QPixmap, QPainter
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QSettings

# Gerekli olması durumunda import et
try:
    from PyQt5.QtChart import QChart, QChartView, QLineSeries, QBarSet, QBarSeries, QPieSeries, QValueAxis, QBarCategoryAxis
    HAS_CHART = True
except ImportError:
    HAS_CHART = False

class BaccaratWorker(QThread):
    """
    Arka planda çalışacak işlemler için QThread sınıfı
    """
    prediction_ready = pyqtSignal(dict)
    operation_complete = pyqtSignal(bool, str)
    progress_update = pyqtSignal(int, str)
    
    def __init__(self, parent=None):
        super(BaccaratWorker, self).__init__(parent)
        self.prediction_engine = None
        self.db_manager = None
        self.performance_tracker = None
        self.operation = None
        self.operation_args = None
    
    def set_components(self, prediction_engine, db_manager, performance_tracker=None):
        """Sistem bileşenlerini ayarlar"""
        self.prediction_engine = prediction_engine
        self.db_manager = db_manager
        self.performance_tracker = performance_tracker
    
    def run(self):
        """Thread başladığında çalışacak metod"""
        if self.operation == "predict":
            try:
                self.progress_update.emit(10, "Veriler hazırlanıyor...")
                prediction = self.prediction_engine.predict(save_prediction=False)
                self.progress_update.emit(100, "Tahmin tamamlandı!")
                self.prediction_ready.emit(prediction)
                self.operation_complete.emit(True, "Tahmin başarıyla hesaplandı.")
            except Exception as e:
                self.operation_complete.emit(False, f"Tahmin hatası: {str(e)}")
        
        elif self.operation == "save_result":
            try:
                result = self.operation_args.get("result")
                self.progress_update.emit(50, f"{result} sonucu kaydediliyor...")
                result_id = self.prediction_engine.update_results(result)
                success = result_id > 0
                self.progress_update.emit(100, "İşlem tamamlandı!")
                self.operation_complete.emit(success, f"Sonuç kaydedildi: {result}" if success else "Sonuç kaydedilemedi!")
            except Exception as e:
                self.operation_complete.emit(False, f"Sonuç kaydetme hatası: {str(e)}")
        
        elif self.operation == "optimize_weights":
            try:
                strategy = self.operation_args.get("strategy", "performance")
                days = self.operation_args.get("days", 7)
                
                self.progress_update.emit(10, "Veriler analiz ediliyor...")
                
                if self.performance_tracker:
                    self.progress_update.emit(40, "Ağırlıklar optimize ediliyor...")
                    updated_weights = self.performance_tracker.optimize_weights()
                    self.progress_update.emit(90, "Değişiklikler kaydediliyor...")
                    
                    # Başarılı optimizasyon
                    self.progress_update.emit(100, "Optimizasyon tamamlandı!")
                    self.operation_complete.emit(True, "Ağırlıklar optimize edildi.")
                else:
                    self.operation_complete.emit(False, "Performance tracker bulunamadı!")
            except Exception as e:
                self.operation_complete.emit(False, f"Ağırlık optimizasyonu hatası: {str(e)}")
        
        elif self.operation == "retrain":
            try:
                model = self.operation_args.get("model")
                epochs = self.operation_args.get("epochs", 50)
                batch_size = self.operation_args.get("batch_size", 32)
                
                if not model:
                    self.operation_complete.emit(False, "Model bulunamadı!")
                    return
                
                from backend.deep_learning.training import ModelTrainer
                model_trainer = ModelTrainer(model, self.db_manager)
                
                self.progress_update.emit(10, "Eğitim verileri hazırlanıyor...")
                
                # Eğitim başlat
                self.progress_update.emit(20, "Model eğitimi başlatılıyor...")
                
                def progress_callback(epoch, total_epochs, metrics):
                    progress = int(20 + (epoch / total_epochs) * 70)
                    self.progress_update.emit(progress, f"Epoch {epoch}/{total_epochs}: Doğruluk={metrics.get('accuracy', 0):.4f}")
                
                # Model eğitimi
                history = model_trainer.train_model(
                    epochs=epochs,
                    batch_size=batch_size,
                    force=True,
                    progress_callback=progress_callback
                )
                
                if history:
                    self.progress_update.emit(100, "Model eğitimi tamamlandı!")
                    self.operation_complete.emit(True, "Model başarıyla eğitildi.")
                else:
                    self.operation_complete.emit(False, "Model eğitimi başarısız oldu!")
            except Exception as e:
                self.operation_complete.emit(False, f"Model eğitim hatası: {str(e)}")
        
        elif self.operation == "bulk_results":
            try:
                results = self.operation_args.get("results", "")
                self.progress_update.emit(10, "Toplu sonuçlar hazırlanıyor...")
                
                valid_results = []
                for char in results:
                    if char in ['P', 'B', 'T']:
                        valid_results.append(char)
                
                total = len(valid_results)
                successful = 0
                
                for i, result in enumerate(valid_results):
                    progress = 10 + int((i / total) * 80)
                    self.progress_update.emit(progress, f"Sonuç kaydediliyor: {result} ({i+1}/{total})")
                    
                    result_id = self.prediction_engine.update_results(result)
                    if result_id > 0:
                        successful += 1
                
                self.progress_update.emit(100, "Toplu sonuç girişi tamamlandı!")
                self.operation_complete.emit(True, f"Toplam {total} sonuçtan {successful} tanesi başarıyla kaydedildi.")
            except Exception as e:
                self.operation_complete.emit(False, f"Toplu sonuç hatası: {str(e)}")


class BaccaratAnalysisApp(QMainWindow):
    def __init__(self, prediction_engine=None, db_manager=None, performance_tracker=None):
        super().__init__()
        self.setWindowTitle("Baccarat Tahmin Analiz Sistemi")
        self.setMinimumSize(800, 600)
        
        # Sistem bileşenleri
        self.prediction_engine = prediction_engine
        self.db_manager = db_manager
        self.performance_tracker = performance_tracker
        
        # Worker thread
        self.worker = BaccaratWorker()
        self.worker.set_components(prediction_engine, db_manager, performance_tracker)
        self.worker.prediction_ready.connect(self.update_prediction_results)
        self.worker.operation_complete.connect(self.handle_operation_complete)
        self.worker.progress_update.connect(self.update_progress)
        
        # UI bileşenleri
        self.prediction_result_label = None
        self.confidence_bar = None
        self.sub_predictions_table = None
        self.status_label = None
        self.progress_bar = None
        self.history_table = None
        
        # Oturum verileri
        self.session_id = None
        self.session_results = []
        
        # Arayüzü oluştur
        self.setup_ui()
        
        # İlk tahmini yap
        if self.prediction_engine and self.db_manager:
            self.get_prediction()
    
    def setup_ui(self):
        """Kullanıcı arayüzünü oluşturur"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C3E50;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1ABC9C;
            }
            QPushButton#analyzeButton {
                background-color: #2ECC71;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
            }
            QPushButton#analyzeButton:hover {
                background-color: #27AE60;
            }
            QTableWidget {
                background-color: #34495E;
                alternate-background-color: #2C3E50;
                color: #ECF0F1;
                gridline-color: #7F8C8D;
                border: 1px solid #7F8C8D;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2C3E50;
                color: #ECF0F1;
                padding: 4px;
                border: 1px solid #7F8C8D;
            }
            QTabWidget::pane {
                border: 1px solid #7F8C8D;
                background-color: #2C3E50;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #34495E;
                color: #ECF0F1;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3498DB;
                color: white;
                font-weight: bold;
            }
            QComboBox {
                background-color: #34495E;
                color: #ECF0F1;
                border: 1px solid #7F8C8D;
                border-radius: 4px;
                padding: 4px;
            }
            QProgressBar {
                border: 1px solid #7F8C8D;
                border-radius: 4px;
                background-color: #34495E;
                color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2ECC71;
                border-radius: 3px;
            }
            QFrame#resultFrame, QGroupBox {
                background-color: #34495E;
                border-radius: 6px;
                padding: 8px;
                border: 1px solid #7F8C8D;
            }
            QGroupBox {
                color: #ECF0F1;
                font-weight: bold;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #34495E;
                color: #ECF0F1;
                border: 1px solid #7F8C8D;
                border-radius: 4px;
                padding: 2px;
            }
            QLineEdit {
                background-color: #34495E;
                color: #ECF0F1;
                border: 1px solid #7F8C8D;
                border-radius: 4px;
                padding: 4px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #7F8C8D;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QCheckBox {
                color: #ECF0F1;
            }
            QRadioButton {
                color: #ECF0F1;
            }
            QTextEdit {
                background-color: #34495E;
                color: #ECF0F1;
                border: 1px solid #7F8C8D;
                border-radius: 4px;
            }
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.create_layout()
    
    def create_layout(self):
        """Ana düzeni oluşturur"""
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Başlık
        header_layout = QHBoxLayout()
        
        header = QLabel("BACCARAT TAHMİN ANALİZ MOTORU")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setStyleSheet("color: #ECF0F1; margin-bottom: 5px;")
        header_layout.addWidget(header)
        
        main_layout.addLayout(header_layout)
        
        # Ana İçerik
        content_layout = QHBoxLayout()
        
        # Sol Panel - Algoritmalar ve Kontroller
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Algoritma Seçimi Bölümü
        self.create_algorithm_section(left_layout)
        
        # Veri Kontrolü Bölümü
        self.create_data_control_section(left_layout)
        
        # Sağ Panel - Analiz Sonuçları
        right_panel = QTabWidget()
        
        # Sonuç Sekmesi
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        self.create_result_tab(result_layout)
        
        # Sonuç Geçmişi Sekmesi
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.create_history_tab(history_layout)
        
        # Grafik Sekmesi
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        self.create_chart_tab(chart_layout)
        
        # Algoritma Ayarları Sekmesi
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        self.create_settings_tab(settings_layout)
        
        # LSTM Modeli Sekmesi
        lstm_tab = QWidget()
        lstm_layout = QVBoxLayout(lstm_tab)
        self.create_lstm_tab(lstm_layout)
        
        # Sekmeleri ekle
        right_panel.addTab(result_tab, "Analiz Sonucu")
        right_panel.addTab(history_tab, "Sonuç Geçmişi")
        right_panel.addTab(chart_tab, "Grafikler")
        right_panel.addTab(settings_tab, "Algoritma Ayarları")
        right_panel.addTab(lstm_tab, "LSTM Modeli")
        
        # Ana İçerik Düzeni
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 2)
        
        main_layout.addLayout(content_layout)
        
        # Alt Menü
        bottom_layout = QHBoxLayout()
        
        # İlerleme Çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - %v")
        bottom_layout.addWidget(self.progress_bar, 2)
        
        # Status
        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setStyleSheet("background-color: #34495E; padding: 5px; border-radius: 3px;")
        bottom_layout.addWidget(self.status_label, 3)
        
        # Butonlar
        export_btn = QPushButton("Dışa Aktar")
        export_btn.clicked.connect(self.export_data)
        
        settings_btn = QPushButton("Genel Ayarlar")
        settings_btn.clicked.connect(self.show_settings)
        
        about_btn = QPushButton("Hakkında")
        about_btn.clicked.connect(self.show_about)
        
        bottom_layout.addWidget(export_btn)
        bottom_layout.addWidget(settings_btn)
        bottom_layout.addWidget(about_btn)
        
        main_layout.addLayout(bottom_layout)
    
    def create_algorithm_section(self, parent_layout):
        """Algoritma seçimi bölümünü oluşturur"""
        algo_group = QGroupBox("Algoritma Seçimi")
        algo_layout = QVBoxLayout(algo_group)
        
        # Algortima listesi
        self.algorithm_combos = []
        self.algorithm_weights = []
        
        # Gerçek algoritmaları al (eğer prediction_engine ayarlanmışsa)
        algorithm_types = []
        if self.prediction_engine and self.prediction_engine.algorithms:
            for algo in self.prediction_engine.algorithms:
                algorithm_types.append(algo.name)
        else:
            # Varsayılan algoritma listesi
            algorithm_types = [
                "Kombinasyon Analizi", 
                "Desen Analizi", 
                "İstatistiksel Model", 
                "Sıralı Desen Analizi", 
                "Bayes Modeli",
                "Markov Zinciri",
                "LSTM"
            ]
        
        for algo in algorithm_types:
            algo_row = QHBoxLayout()
            
            # Seçim kutusu
            check = QCheckBox(algo)
            check.setChecked(True)
            check.stateChanged.connect(self.algorithm_selection_changed)
            algo_row.addWidget(check, 3)
            
            # Ağırlık ayarı
            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.1, 5.0)
            weight_spin.setSingleStep(0.1)
            weight_spin.setValue(1.0)
            weight_spin.setPrefix("Ağırlık: ")
            weight_spin.valueChanged.connect(self.algorithm_weight_changed)
            algo_row.addWidget(weight_spin, 2)
            
            algo_layout.addLayout(algo_row)
            self.algorithm_combos.append(check)
            self.algorithm_weights.append(weight_spin)
        
        parent_layout.addWidget(algo_group)
    
    def create_data_control_section(self, parent_layout):
        """Veri kontrolü bölümünü oluşturur"""
        data_group = QGroupBox("Veri Kontrolü")
        data_layout = QVBoxLayout(data_group)
        
        # Veri Kaynağı
        data_source_layout = QHBoxLayout()
        data_source_layout.addWidget(QLabel("Veri Kaynağı:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Son 50 Sonuç", "Son 100 Sonuç", "Son 200 Sonuç", "Tüm Veri"])
        data_source_layout.addWidget(self.data_source_combo)
        data_layout.addLayout(data_source_layout)
        
        # Pencere Boyutu Ayarı
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Pencere Boyutu:"))
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(5, 50)
        self.window_size_spin.setValue(10)
        window_layout.addWidget(self.window_size_spin)
        data_layout.addLayout(window_layout)
        
        # Sonuç Girişi
        result_input_layout = QHBoxLayout()
        result_input_layout.addWidget(QLabel("Sonuç Girişi:"))
        
        result_btn_layout = QHBoxLayout()
        player_btn = QPushButton("P")
        banker_btn = QPushButton("B")
        tie_btn = QPushButton("T")
        
        player_btn.setMinimumWidth(40)
        banker_btn.setMinimumWidth(40)
        tie_btn.setMinimumWidth(40)
        
        # Renk stilleri
        player_btn.setStyleSheet("background-color: #3498DB;")
        banker_btn.setStyleSheet("background-color: #E74C3C;")
        tie_btn.setStyleSheet("background-color: #2ECC71;")
        
        player_btn.clicked.connect(lambda: self.enter_result('P'))
        banker_btn.clicked.connect(lambda: self.enter_result('B'))
        tie_btn.clicked.connect(lambda: self.enter_result('T'))
        
        result_btn_layout.addWidget(player_btn)
        result_btn_layout.addWidget(banker_btn)
        result_btn_layout.addWidget(tie_btn)
        
        result_input_layout.addLayout(result_btn_layout)
        data_layout.addLayout(result_input_layout)
        
        # Toplu Sonuç Girişi
        bulk_input_layout = QHBoxLayout()
        bulk_input_layout.addWidget(QLabel("Toplu Giriş:"))
        
        self.bulk_input = QLineEdit()
        self.bulk_input.setPlaceholderText("Örn: PPBBPTPBB")
        bulk_input_layout.addWidget(self.bulk_input)
        
        bulk_btn = QPushButton("Gir")
        bulk_btn.clicked.connect(self.enter_bulk_results)
        bulk_input_layout.addWidget(bulk_btn)
        
        data_layout.addLayout(bulk_input_layout)
        
        # Analiz Butonu
        analyze_btn = QPushButton("ANALİZ ET")
        analyze_btn.setObjectName("analyzeButton")
        analyze_btn.clicked.connect(self.get_prediction)
        data_layout.addWidget(analyze_btn)
        
        # Ağırlık Optimizasyonu Butonu
        optimize_btn = QPushButton("Ağırlıkları Optimize Et")
        optimize_btn.clicked.connect(self.optimize_weights)
        data_layout.addWidget(optimize_btn)
        
        parent_layout.addWidget(data_group)
    
    def create_result_tab(self, parent_layout):
        """Sonuç sekmesini oluşturur"""
        # Tahmin Özeti
        prediction_frame = QFrame()
        prediction_frame.setObjectName("resultFrame")
        prediction_layout = QVBoxLayout(prediction_frame)
        
        prediction_header = QLabel("TAHMİN SONUCU")
        prediction_header.setFont(QFont("Arial", 14, QFont.Bold))
        prediction_header.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(prediction_header)
        
        # Ana tahmin
        self.prediction_result_label = QLabel("...")
        self.prediction_result_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.prediction_result_label.setStyleSheet("color: #F39C12;")
        self.prediction_result_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.prediction_result_label)
        
        # Güven seviyesi
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Güven Seviyesi:"))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setValue(0)
        confidence_layout.addWidget(self.confidence_bar)
        prediction_layout.addLayout(confidence_layout)
        
        # Alt tahminler
        sub_predictions_label = QLabel("Alt Algoritma Tahminleri:")
        prediction_layout.addWidget(sub_predictions_label)
        
        self.sub_predictions_table = QTableWidget(0, 3)
        self.sub_predictions_table.setHorizontalHeaderLabels(["Algoritma", "Tahmin", "Güven"])
        self.sub_predictions_table.horizontalHeader().setStretchLastSection(True)
        
        prediction_layout.addWidget(self.sub_predictions_table)
        
        parent_layout.addWidget(prediction_frame)
        
        # İstatistikler
        stats_frame = QFrame()
        stats_frame.setObjectName("resultFrame")
        stats_layout = QVBoxLayout(stats_frame)
        
        stats_header = QLabel("İSTATİSTİKLER")
        stats_header.setFont(QFont("Arial", 12, QFont.Bold))
        stats_header.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_header)
        
        # İstatistik grid
        stats_grid = QGridLayout()
        
        # İstatistik değerleri için QLabel'lar
        self.stats_labels = {}
        
        stats = [
            ("Toplam Tahmin", "total_predictions", "0"),
            ("Doğru Tahmin", "correct_predictions", "0"),
            ("Başarı Oranı", "accuracy", "0%"),
            ("En İyi Algoritma", "best_algorithm", "-"),
            ("Art Arda Banker", "banker_streak", "0"),
            ("Art Arda Player", "player_streak", "0")
        ]
        
        for i, (stat, key, val) in enumerate(stats):
            row = i // 2
            col = i % 2 * 2
            
            stat_label = QLabel(stat + ":")
            val_label = QLabel(val)
            val_label.setStyleSheet("color: #F39C12; font-weight: bold;")
            
            self.stats_labels[key] = val_label
            
            stats_grid.addWidget(stat_label, row, col)
            stats_grid.addWidget(val_label, row, col + 1)
        
        stats_layout.addLayout(stats_grid)
        parent_layout.addWidget(stats_frame)
        
        # Veri kaynağını düzenli güncelle
        self.update_statistics()
    
    def create_history_tab(self, parent_layout):
        """Sonuç geçmişi sekmesini oluşturur"""
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["#", "Zaman", "Sonuç", "Tahmin", "Doğru"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setAlternatingRowColors(True)
        
        parent_layout.addWidget(self.history_table)
        
        # Geçmiş sonuçları yükle
        self.load_history()
    
    def create_chart_tab(self, parent_layout):
        """Grafik sekmesini oluşturur"""
        if HAS_CHART:
            # Dağılım grafiği
            distribution_group = QGroupBox("Sonuç Dağılımı")
            distribution_layout = QVBoxLayout(distribution_group)
            
            # Pasta grafiği oluştur
            pie_series = QPieSeries()
            
            if self.db_manager:
                # Verileri al
                results = self.db_manager.get_all_results()
                
                if results:
                    # Sonuçları say
                    p_count = results.count('P')
                    b_count = results.count('B')
                    t_count = results.count('T')
                    
                    # Pasta dilimlerini ekle
                    if p_count > 0:
                        slice_p = pie_series.append("Player", p_count)
                        slice_p.setBrush(QColor("#3498DB"))
                    
                    if b_count > 0:
                        slice_b = pie_series.append("Banker", b_count)
                        slice_b.setBrush(QColor("#E74C3C"))
                    
                    if t_count > 0:
                        slice_t = pie_series.append("Tie", t_count)
                        slice_t.setBrush(QColor("#2ECC71"))
            
            # Eğer veri yoksa, örnek veriler ekle
            if not self.db_manager or len(pie_series) == 0:
                slice_p = pie_series.append("Player", 45)
                slice_p.setBrush(QColor("#3498DB"))
                
                slice_b = pie_series.append("Banker", 50)
                slice_b.setBrush(QColor("#E74C3C"))
                
                slice_t = pie_series.append("Tie", 5)
                slice_t.setBrush(QColor("#2ECC71"))
            
            # Grafik oluştur
            chart = QChart()
            chart.addSeries(pie_series)
            chart.setTitle("Sonuç Dağılımı")
            chart.setTheme(QChart.ChartThemeDark)
            chart.legend().setVisible(True)
            
            # Grafik görünümü
            chart_view = QChartView(chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            
            distribution_layout.addWidget(chart_view)
            
            # Doğruluk grafiği
            accuracy_group = QGroupBox("Algoritma Performansı")
            accuracy_layout = QVBoxLayout(accuracy_group)
            
            # Bar grafiği oluştur
            bar_set = QBarSet("Doğruluk")
            bar_set.setColor(QColor("#F39C12"))
            
            categories = []
            
            if self.prediction_engine and self.prediction_engine.algorithms:
                # Gerçek algoritma verilerini kullan
                for algo in self.prediction_engine.algorithms:
                    categories.append(algo.name)
                    bar_set.append(algo.accuracy * 100)  # Yüzde olarak doğruluk
            else:
                # Örnek veriler
                categories = ["Desen", "İstatistik", "Bayes", "Markov", "LSTM"]
                sample_accuracies = [68, 72, 65, 70, 74]
                for acc in sample_accuracies:
                    bar_set.append(acc)
            
            # Bar serisini oluştur
            bar_series = QBarSeries()
            bar_series.append(bar_set)
            
            # Grafik oluştur
            accuracy_chart = QChart()
            accuracy_chart.addSeries(bar_series)
            accuracy_chart.setTitle("Algoritma Doğruluk Oranları")
            accuracy_chart.setTheme(QChart.ChartThemeDark)
            
            # Eksenler
            axis_x = QBarCategoryAxis()
            axis_x.append(categories)
            accuracy_chart.addAxis(axis_x, Qt.AlignBottom)
            bar_series.attachAxis(axis_x)
            
            axis_y = QValueAxis()
            axis_y.setRange(0, 100)
            axis_y.setTitleText("Doğruluk (%)")
            accuracy_chart.addAxis(axis_y, Qt.AlignLeft)
            bar_series.attachAxis(axis_y)
            
            accuracy_chart.legend().setVisible(False)
            
            # Grafik görünümü
            accuracy_view = QChartView(accuracy_chart)
            accuracy_view.setRenderHint(QPainter.Antialiasing)
            
            accuracy_layout.addWidget(accuracy_view)
            
            # Grafikleri düzene ekle
            parent_layout.addWidget(distribution_group)
            parent_layout.addWidget(accuracy_group)
        else:
            # PyQtChart yoksa, basit bir mesaj göster
            chart_placeholder = QLabel("Grafik gösterimi için PyQtChart modülü gereklidir.\npip install PyQtChart komutunu çalıştırın.")
            chart_placeholder.setAlignment(Qt.AlignCenter)
            chart_placeholder.setStyleSheet("background-color: #34495E; padding: 80px; border-radius: 4px;")
            parent_layout.addWidget(chart_placeholder)
    
    def create_settings_tab(self, parent_layout):
        """Algoritma ayarları sekmesini oluşturur"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Kombinasyon Analizi Ayarları
        combo_settings = QGroupBox("Kombinasyon Analizi Ayarları")
        combo_layout = QVBoxLayout(combo_settings)
        
        combo_layout.addWidget(QLabel("Alt algoritmaların ağırlıklarını ayarlayın:"))
        
        self.algo_sliders = {}
        
        if self.prediction_engine and self.prediction_engine.algorithms:
            # Gerçek algoritmaları al
            algorithms = []
            for algo in self.prediction_engine.algorithms:
                if algo.name != "Combination Analysis":
                    algorithms.append(algo.name)
        else:
            # Örnek algoritmalar
            algorithms = ["Desen Analizi", "İstatistiksel Model", "Sıralı Desen", "Bayes Modeli", "Markov Zinciri"]
        
        for i, algo in enumerate(algorithms):
            algo_weight_layout = QHBoxLayout()
            algo_weight_layout.addWidget(QLabel(algo + ":"))
            
            algo_slider = QSlider(Qt.Horizontal)
            algo_slider.setRange(1, 30)
            algo_slider.setValue(10)
            algo_slider.setFixedWidth(150)
            
            weight_label = QLabel("1.0")
            
            # Slider değiştiğinde etiketi güncelle
            algo_slider.valueChanged.connect(lambda v, label=weight_label: label.setText(f"{v/10:.1f}"))
            
            self.algo_sliders[algo] = (algo_slider, weight_label)
            
            algo_weight_layout.addWidget(algo_slider)
            algo_weight_layout.addWidget(weight_label)
            combo_layout.addLayout(algo_weight_layout)
        
        scroll_layout.addWidget(combo_settings)
        
        # Destek Vektör Analizi Ayarları
        pattern_settings = QGroupBox("Desen Analizi Ayarları")
        pattern_layout = QVBoxLayout(pattern_settings)
        
        # Desen uzunluğu
        pattern_length_layout = QHBoxLayout()
        pattern_length_layout.addWidget(QLabel("Desen Uzunluğu:"))
        pattern_length = QSpinBox()
        pattern_length.setRange(1, 10)
        pattern_length.setValue(3)
        pattern_length_layout.addWidget(pattern_length)
        pattern_layout.addLayout(pattern_length_layout)
        
        # Minimum örnek sayısı
        min_samples_layout = QHBoxLayout()
        min_samples_layout.addWidget(QLabel("Minimum Örnek Sayısı:"))
        min_samples = QSpinBox()
        min_samples.setRange(1, 50)
        min_samples.setValue(5)
        min_samples_layout.addWidget(min_samples)
        pattern_layout.addLayout(min_samples_layout)
        
        scroll_layout.addWidget(pattern_settings)
        
        # Markov Zinciri Ayarları
        markov_settings = QGroupBox("Markov Zinciri Ayarları")
        markov_layout = QVBoxLayout(markov_settings)
        
        # Zincir mertebesi
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Zincir Mertebesi:"))
        markov_order = QSpinBox()
        markov_order.setRange(1, 5)
        markov_order.setValue(2)
        order_layout.addWidget(markov_order)
        markov_layout.addLayout(order_layout)
        
        scroll_layout.addWidget(markov_settings)
        
        # Monte Carlo Simülasyonu Ayarları
        monte_carlo_settings = QGroupBox("Monte Carlo Simülasyonu Ayarları")
        monte_carlo_layout = QVBoxLayout(monte_carlo_settings)
        
        # Simülasyon sayısı
        simulations_layout = QHBoxLayout()
        simulations_layout.addWidget(QLabel("Simülasyon Sayısı:"))
        simulations = QSpinBox()
        simulations.setRange(100, 10000)
        simulations.setSingleStep(100)
        simulations.setValue(1000)
        simulations_layout.addWidget(simulations)
        monte_carlo_layout.addLayout(simulations_layout)
        
        scroll_layout.addWidget(monte_carlo_settings)
        
        # Ayarları Uygula butonu
        apply_settings_btn = QPushButton("Ayarları Uygula")
        apply_settings_btn.clicked.connect(self.apply_algorithm_settings)
        scroll_layout.addWidget(apply_settings_btn)
        
        # Boşluk ekle
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        parent_layout.addWidget(scroll_area)
    
    def create_lstm_tab(self, parent_layout):
        """LSTM modeli sekmesini oluşturur"""
        lstm_group = QGroupBox("LSTM Derin Öğrenme Modeli")
        lstm_layout = QVBoxLayout(lstm_group)
        
        # Model Durumu
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("Model Durumu:"), 0, 0)
        self.lstm_status_label = QLabel("Yüklü" if self.prediction_engine and self.prediction_engine.deep_learning_model else "Yüklü Değil")
        self.lstm_status_label.setStyleSheet("color: #2ECC71;" if self.prediction_engine and self.prediction_engine.deep_learning_model else "color: #E74C3C;")
        status_layout.addWidget(self.lstm_status_label, 0, 1)
        
        status_layout.addWidget(QLabel("Doğruluk Oranı:"), 1, 0)
        self.lstm_accuracy_label = QLabel("0%")
        status_layout.addWidget(self.lstm_accuracy_label, 1, 1)
        
        status_layout.addWidget(QLabel("Eğitim Durumu:"), 2, 0)
        self.lstm_trained_label = QLabel("Eğitilmiş" if self.prediction_engine and self.prediction_engine.deep_learning_model and hasattr(self.prediction_engine.deep_learning_model, 'trained') and self.prediction_engine.deep_learning_model.trained else "Eğitilmemiş")
        self.lstm_trained_label.setStyleSheet("color: #2ECC71;" if self.prediction_engine and self.prediction_engine.deep_learning_model and hasattr(self.prediction_engine.deep_learning_model, 'trained') and self.prediction_engine.deep_learning_model.trained else "color: #E74C3C;")
        status_layout.addWidget(self.lstm_trained_label, 2, 1)
        
        lstm_layout.addLayout(status_layout)
        
        # Eğitim Ayarları
        training_group = QGroupBox("Eğitim Ayarları")
        training_layout = QVBoxLayout(training_group)
        
        # Epoch sayısı
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epoch Sayısı:"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(10, 1000)
        self.epoch_spin.setSingleStep(10)
        self.epoch_spin.setValue(50)
        epoch_layout.addWidget(self.epoch_spin)
        training_layout.addLayout(epoch_layout)
        
        # Batch boyutu
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Boyutu:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 256)
        self.batch_spin.setSingleStep(8)
        self.batch_spin.setValue(32)
        batch_layout.addWidget(self.batch_spin)
        training_layout.addLayout(batch_layout)
        
        # Eğitim butonu
        train_btn = QPushButton("Modeli Eğit")
        train_btn.clicked.connect(self.train_lstm_model)
        training_layout.addWidget(train_btn)
        
        lstm_layout.addWidget(training_group)
        
        # Model Performansı
        performance_group = QGroupBox("Model Performansı")
        performance_layout = QVBoxLayout(performance_group)
        
        # Son 50 tahminde doğruluk
        recent_layout = QHBoxLayout()
        recent_layout.addWidget(QLabel("Son 50 Tahminde Doğruluk:"))
        self.lstm_recent_bar = QProgressBar()
        self.lstm_recent_bar.setValue(0)
        recent_layout.addWidget(self.lstm_recent_bar)
        performance_layout.addLayout(recent_layout)
        
        # Grafik alanı
        # Basit bir placeholder - gerçek uygulamada bir grafik eklenebilir
        history_placeholder = QLabel("LSTM Eğitim Geçmişi Grafiği")
        history_placeholder.setAlignment(Qt.AlignCenter)
        history_placeholder.setStyleSheet("background-color: #2C3E50; padding: 40px; border-radius: 4px;")
        performance_layout.addWidget(history_placeholder)
        
        lstm_layout.addWidget(performance_group)
        
        # LSTM bilgilerini güncelle
        self.update_lstm_info()
        
        parent_layout.addWidget(lstm_group)
    
    def update_lstm_info(self):
        """LSTM model bilgilerini güncelle"""
        if not self.prediction_engine or not self.prediction_engine.deep_learning_model:
            return
        
        lstm_model = self.prediction_engine.deep_learning_model
        
        # Model durumu
        self.lstm_status_label.setText("Yüklü")
        self.lstm_status_label.setStyleSheet("color: #2ECC71;")
        
        # Eğitim durumu
        is_trained = hasattr(lstm_model, 'trained') and lstm_model.trained
        self.lstm_trained_label.setText("Eğitilmiş" if is_trained else "Eğitilmemiş")
        self.lstm_trained_label.setStyleSheet("color: #2ECC71;" if is_trained else "color: #E74C3C;")
        
        # Doğruluk oranı
        accuracy = 0.0
        if hasattr(lstm_model, 'accuracy'):
            accuracy = lstm_model.accuracy
        elif hasattr(lstm_model, 'training_history') and lstm_model.training_history:
            if 'accuracy' in lstm_model.training_history:
                accuracy_history = lstm_model.training_history['accuracy']
                if accuracy_history:
                    accuracy = accuracy_history[-1]
        
        self.lstm_accuracy_label.setText(f"{accuracy:.2%}")
        
        # Son performans
        if self.db_manager:
            algorithm_info = self.db_manager.get_algorithm_by_name("LSTM")
            if algorithm_info:
                recent_accuracy = algorithm_info.get('current_accuracy', 0.0)
                self.lstm_recent_bar.setValue(int(recent_accuracy * 100))
    
    def algorithm_selection_changed(self):
        """Algoritma seçimi değiştiğinde çağrılır"""
        if not self.prediction_engine:
            return
        
        for i, check in enumerate(self.algorithm_combos):
            if i < len(self.prediction_engine.algorithms):
                algorithm = self.prediction_engine.algorithms[i]
                enabled = check.isChecked()
                
                # Algoritmanın etkinliğini değiştir
                # Gerçek uygulamada burada prediction_engine'de bir yöntem olmalı
                # Şimdilik basit bir log mesajı
                print(f"Algoritma {algorithm.name} {'etkin' if enabled else 'devre dışı'}")
    
    def algorithm_weight_changed(self):
        """Algoritma ağırlığı değiştiğinde çağrılır"""
        if not self.prediction_engine:
            return
        
        for i, spin in enumerate(self.algorithm_weights):
            if i < len(self.prediction_engine.algorithms):
                algorithm = self.prediction_engine.algorithms[i]
                weight = spin.value()
                
                # Algoritmanın ağırlığını değiştir
                algorithm.set_weight(weight)
                print(f"Algoritma {algorithm.name} ağırlığı: {weight}")
    
    def apply_algorithm_settings(self):
        """Algoritma ayarlarını uygular"""
        # Burada gerçek algoritma nesnelerine ayarlar uygulanmalı
        # Basit bir gösterim için sadece bir mesaj gösteriyoruz
        QMessageBox.information(self, "Ayarlar", "Algoritma ayarları uygulandı.")
    
    def get_prediction(self):
        """Tahmin yap"""
        if not self.prediction_engine or not self.db_manager:
            QMessageBox.warning(self, "Hata", "Tahmin motoru veya veritabanı bağlantısı bulunamadı!")
            return
        
        # Veri kaynağını belirle
        data_source = self.data_source_combo.currentText()
        limit = 50
        
        if data_source == "Son 100 Sonuç":
            limit = 100
        elif data_source == "Son 200 Sonuç":
            limit = 200
        elif data_source == "Tüm Veri":
            limit = 1000  # Çok büyük bir sayı
        
        # Pencere boyutu
        window_size = self.window_size_spin.value()
        
        # Durum bilgisini güncelle
        self.status_label.setText("Durum: Tahmin hesaplanıyor...")
        self.progress_bar.setValue(10)
        
        # Worker thread'i kullanarak tahmini hesapla
        self.worker.operation = "predict"
        self.worker.start()
    
    def update_prediction_results(self, prediction):
        """Tahmin sonuçlarını güncelle"""
        if not prediction:
            return
        
        # Ana tahmin
        result = prediction.get('prediction', '')
        confidence = prediction.get('confidence', 0.0)
        
        # Result etiketi renk
        result_color = "#3498DB"  # Player (mavi)
        if result == 'B':
            result_color = "#E74C3C"  # Banker (kırmızı)
        elif result == 'T':
            result_color = "#2ECC71"  # Tie (yeşil)
        
        self.prediction_result_label.setText(result)
        self.prediction_result_label.setStyleSheet(f"color: {result_color}; font-weight: bold;")
        
        # Güven çubuğunu güncelle
        self.confidence_bar.setValue(int(confidence * 100))
        
        # Alt tahminleri güncelle
        algorithms = prediction.get('algorithms', [])
        self.sub_predictions_table.setRowCount(len(algorithms))
        
        for i, algo in enumerate(algorithms):
            algo_name = algo.get('algorithm', '')
            algo_pred = algo.get('prediction', '')
            algo_conf = algo.get('confidence', 0.0)
            
            # Tablo öğeleri
            name_item = QTableWidgetItem(algo_name)
            pred_item = QTableWidgetItem(algo_pred)
            conf_item = QTableWidgetItem(f"{algo_conf:.2%}")
            
            # Renklendirme
            if algo_pred == result:
                pred_item.setForeground(QColor(result_color))
                pred_item.setFont(QFont("Arial", 9, QFont.Bold))
            
            # Tabloyu güncelle
            self.sub_predictions_table.setItem(i, 0, name_item)
            self.sub_predictions_table.setItem(i, 1, pred_item)
            self.sub_predictions_table.setItem(i, 2, conf_item)
        
        # Durum bilgisi güncelle
        self.status_label.setText("Durum: Tahmin hazır")
        self.progress_bar.setValue(100)
        
        # İstatistikleri güncelle
        self.update_statistics()
    
    def enter_result(self, result):
        """Yeni bir sonuç gir"""
        if not self.prediction_engine or not self.db_manager:
            QMessageBox.warning(self, "Hata", "Tahmin motoru veya veritabanı bağlantısı bulunamadı!")
            return
        
        # Durum bilgisini güncelle
        self.status_label.setText(f"Durum: {result} sonucu kaydediliyor...")
        self.progress_bar.setValue(0)
        
        # Worker thread'i kullanarak sonucu kaydet
        self.worker.operation = "save_result"
        self.worker.operation_args = {"result": result}
        self.worker.start()
    
    def enter_bulk_results(self):
        """Toplu sonuç gir"""
        if not self.prediction_engine or not self.db_manager:
            QMessageBox.warning(self, "Hata", "Tahmin motoru veya veritabanı bağlantısı bulunamadı!")
            return
        
        # Giriş alanındaki metni al
        bulk_text = self.bulk_input.text().upper()
        
        if not bulk_text:
            QMessageBox.warning(self, "Hata", "Lütfen sonuçları girin!")
            return
        
        # Durum bilgisini güncelle
        self.status_label.setText("Durum: Toplu sonuçlar işleniyor...")
        self.progress_bar.setValue(0)
        
        # Worker thread'i kullanarak sonuçları kaydet
        self.worker.operation = "bulk_results"
        self.worker.operation_args = {"results": bulk_text}
        self.worker.start()
    
    def optimize_weights(self):
        """Algoritma ağırlıklarını optimize et"""
        if not self.performance_tracker:
            QMessageBox.warning(self, "Hata", "Performans izleyici bulunamadı!")
            return
        
        # Kullanıcıya sorma
        reply = QMessageBox.question(
            self, 
            "Ağırlık Optimizasyonu", 
            "Algoritma ağırlıkları son performans verilerine göre optimize edilecek. Devam etmek istiyor musunuz?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return
        
        # Durum bilgisini güncelle
        self.status_label.setText("Durum: Ağırlıklar optimize ediliyor...")
        self.progress_bar.setValue(0)
        
        # Worker thread'i kullanarak ağırlıkları optimize et
        self.worker.operation = "optimize_weights"
        self.worker.operation_args = {
            "strategy": "performance",
            "days": 7,
            "performance_tracker": self.performance_tracker
        }
        self.worker.start()
    
    def train_lstm_model(self):
        """LSTM modelini eğit"""
        if not self.prediction_engine or not self.prediction_engine.deep_learning_model or not self.db_manager:
            QMessageBox.warning(self, "Hata", "LSTM modeli veya veritabanı bağlantısı bulunamadı!")
            return
        
        # Son kayıt sayısını kontrol et
        all_results = self.db_manager.get_all_results()
        if len(all_results) < 200:
            QMessageBox.warning(self, "Hata", "Yeterli veri yok! En az 200 sonuç gerekli.")
            return
        
        # Kullanıcıya sorma
        reply = QMessageBox.question(
            self, 
            "LSTM Eğitimi", 
            f"LSTM modeli {self.epoch_spin.value()} epoch boyunca eğitilecek. Bu işlem biraz zaman alabilir. Devam etmek istiyor musunuz?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return
        
        # Durum bilgisini güncelle
        self.status_label.setText("Durum: LSTM modeli eğitiliyor...")
        self.progress_bar.setValue(0)
        
        # Worker thread'i kullanarak modeli eğit
        self.worker.operation = "retrain"
        self.worker.operation_args = {
            "model": self.prediction_engine.deep_learning_model,
            "epochs": self.epoch_spin.value(),
            "batch_size": self.batch_spin.value()
        }
        self.worker.start()
    
    def handle_operation_complete(self, success, message):
        """İşlem tamamlandığında çağrılır"""
        if success:
            # Başarılı işlem - yeni tahmin al
            self.status_label.setText(f"Durum: {message}")
            self.get_prediction()
            
            # Geçmişi güncelle
            self.load_history()
            
            # LSTM bilgilerini güncelle
            self.update_lstm_info()
        else:
            # Başarısız işlem
            self.status_label.setText(f"Durum: {message}")
            QMessageBox.warning(self, "Hata", message)
    
    def update_progress(self, value, message):
        """İlerleme çubuğunu günceller"""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Durum: {message}")
    
    def update_statistics(self):
        """İstatistik bilgilerini güncelle"""
        if not self.db_manager:
            return
        
        try:
            # Toplam tahmin sayısı
            total_predictions = 0
            correct_predictions = 0
            
            # Algoritma performans verilerini al
            if self.prediction_engine and self.prediction_engine.algorithms:
                best_algorithm = None
                best_accuracy = 0
                
                for algorithm in self.prediction_engine.algorithms:
                    if algorithm.accuracy > best_accuracy:
                        best_accuracy = algorithm.accuracy
                        best_algorithm = algorithm.name
                    
                    total_predictions += algorithm.total_predictions
                    correct_predictions += algorithm.correct_predictions
                
                # İstatistik etiketlerini güncelle
                self.stats_labels["total_predictions"].setText(str(total_predictions))
                self.stats_labels["correct_predictions"].setText(str(correct_predictions))
                
                if total_predictions > 0:
                    accuracy = correct_predictions / total_predictions
                    self.stats_labels["accuracy"].setText(f"{accuracy:.2%}")
                
                if best_algorithm:
                    self.stats_labels["best_algorithm"].setText(best_algorithm)
            
            # Seri uzunluklarını hesapla
            all_results = self.db_manager.get_all_results()
            
            if all_results:
                # Banker serisi
                banker_streak = 0
                max_banker_streak = 0
                
                # Player serisi
                player_streak = 0
                max_player_streak = 0
                
                # Serileri hesapla
                for result in all_results:
                    if result == 'B':
                        banker_streak += 1
                        player_streak = 0
                        max_banker_streak = max(max_banker_streak, banker_streak)
                    elif result == 'P':
                        player_streak += 1
                        banker_streak = 0
                        max_player_streak = max(max_player_streak, player_streak)
                    else:  # Tie
                        banker_streak = 0
                        player_streak = 0
                
                # Seri etiketlerini güncelle
                self.stats_labels["banker_streak"].setText(str(max_banker_streak))
                self.stats_labels["player_streak"].setText(str(max_player_streak))
        
        except Exception as e:
            print(f"İstatistik güncelleme hatası: {str(e)}")
    
    def load_history(self):
        """Geçmiş sonuçları yükle"""
        if not self.db_manager:
            return
        
        try:
            # Son 50 sonucu al
            results = self.db_manager.get_last_n_results(50)
            
            if not results:
                return
            
            # Tabloyu temizle
            self.history_table.setRowCount(len(results))
            
            # Sonuçları tersine çevir (en yeniden en eskiye)
            results.reverse()
            
            for i, result in enumerate(results):
                result_id = result['id']
                timestamp = result['timestamp']
                result_value = result['result']
                
                # Tahminleri al - gerçek uygulamada DB'den alınmalı
                prediction = None
                is_correct = False
                
                # Tablo öğeleri
                id_item = QTableWidgetItem(str(result_id))
                time_item = QTableWidgetItem(timestamp)
                result_item = QTableWidgetItem(result_value)
                
                # Sonuç renklendirme
                if result_value == 'P':
                    result_item.setForeground(QColor("#3498DB"))
                elif result_value == 'B':
                    result_item.setForeground(QColor("#E74C3C"))
                else:  # Tie
                    result_item.setForeground(QColor("#2ECC71"))
                
                # Tahmin ve doğruluk öğeleri
                pred_item = QTableWidgetItem("-" if prediction is None else prediction)
                correct_item = QTableWidgetItem("✓" if is_correct else "✗")
                
                if is_correct:
                    correct_item.setForeground(QColor("#2ECC71"))
                else:
                    correct_item.setForeground(QColor("#E74C3C"))
                
                # Tabloyu güncelle
                self.history_table.setItem(i, 0, id_item)
                self.history_table.setItem(i, 1, time_item)
                self.history_table.setItem(i, 2, result_item)
                self.history_table.setItem(i, 3, pred_item)
                self.history_table.setItem(i, 4, correct_item)
        
        except Exception as e:
            print(f"Geçmiş yükleme hatası: {str(e)}")
    
    def export_data(self):
        """Verileri dışa aktar"""
        if not self.db_manager:
            QMessageBox.warning(self, "Hata", "Veritabanı bağlantısı bulunamadı!")
            return
        
        # Dosya adını kullanıcıdan al
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            "Verileri Dışa Aktar", 
            f"baccarat_export_{datetime.now().strftime('%Y%m%d')}.csv", 
            "CSV Dosyaları (*.csv)"
        )
        
        if not file_name:
            return
        
        try:
            # Tüm sonuçları al
            results = self.db_manager.get_all_results()
            
            if not results:
                QMessageBox.warning(self, "Hata", "Dışa aktarılacak veri bulunamadı!")
                return
            
            # CSV dosyasını oluştur
            with open(file_name, 'w') as f:
                f.write("ID,Tarih,Sonuç\n")
                
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        result_id = result.get('id', i+1)
                        timestamp = result.get('timestamp', '')
                        result_value = result.get('result', '')
                    else:
                        result_id = i+1
                        timestamp = ''
                        result_value = result
                    
                    f.write(f"{result_id},{timestamp},{result_value}\n")
            
            QMessageBox.information(self, "Başarılı", f"Veriler başarıyla dışa aktarıldı: {file_name}")
        
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri dışa aktarma hatası: {str(e)}")
    
    def show_settings(self):
        """Genel ayarlar penceresi"""
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Genel Ayarlar")
        settings_dialog.setMinimumWidth(400)
        settings_dialog.setStyleSheet("""
            QDialog {
                background-color: #2C3E50;
            }
            QLabel {
                color: #ECF0F1;
            }
            QGroupBox {
                color: #ECF0F1;
                font-weight: bold;
                border: 1px solid #7F8C8D;
                border-radius: 4px;
                margin-top: 2ex;
                padding: 10px;
            }
            QCheckBox {
                color: #ECF0F1;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout(settings_dialog)
        
        # Veritabanı ayarları
        db_group = QGroupBox("Veritabanı Ayarları")
        db_layout = QFormLayout(db_group)
        
        db_path = QLineEdit()
        db_path.setText(self.db_manager.db_uri if self.db_manager else "baccarat.db")
        db_layout.addRow("Veritabanı Yolu:", db_path)
        
        reset_db_btn = QPushButton("Veritabanını Sıfırla")
        reset_db_btn.setStyleSheet("background-color: #E74C3C;")
        db_layout.addRow("", reset_db_btn)
        
        layout.addWidget(db_group)
        
        # Uygulama ayarları
        app_group = QGroupBox("Uygulama Ayarları")
        app_layout = QFormLayout(app_group)
        
        dark_mode = QCheckBox("Koyu Tema")
        dark_mode.setChecked(True)
        app_layout.addRow("", dark_mode)
        
        debug_mode = QCheckBox("Debug Modu")
        debug_mode.setChecked(False)
        app_layout.addRow("", debug_mode)
        
        layout.addWidget(app_group)
        
        # Butonlar
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Kaydet")
        cancel_btn = QPushButton("İptal")
        
        save_btn.clicked.connect(settings_dialog.accept)
        cancel_btn.clicked.connect(settings_dialog.reject)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        settings_dialog.exec_()
    
    def show_about(self):
        """Hakkında penceresi"""
        about_text = """
        <h2>Baccarat Tahmin Analiz Sistemi</h2>
        <p>Sürüm 1.0</p>
        <p>14 farklı algoritma ve derin öğrenme entegrasyonu içeren 
        Baccarat tahmin yazılımı.</p>
        <p>Bu uygulama PyQt5 kullanılarak oluşturulmuştur.</p>
        <p>© 2023 Tüm hakları saklıdır.</p>
        """
        
        QMessageBox.about(self, "Hakkında", about_text)