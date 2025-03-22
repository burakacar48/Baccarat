#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyQt5 arayüzü hata düzeltme scripti
-----------------------------------
qt_ui.py dosyasındaki hataları ve eksikleri tespit eder ve düzeltir.
"""

import os
import sys
import re
import shutil
from datetime import datetime

def fix_qt_ui_file():
    """
    qt_ui.py dosyasındaki hataları ve eksikleri düzeltir
    """
    # Dosya yolu
    frontend_dir = os.path.join(os.getcwd(), 'frontend')
    qt_ui_path = os.path.join(frontend_dir, 'qt_ui.py')
    
    # Dosyanın var olup olmadığını kontrol et
    if not os.path.exists(qt_ui_path):
        print(f"Hata: {qt_ui_path} bulunamadı!")
        return False
    
    # Yedek oluştur
    backup_path = os.path.join(frontend_dir, f"qt_ui_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
    try:
        shutil.copy2(qt_ui_path, backup_path)
        print(f"Yedek dosya oluşturuldu: {backup_path}")
    except Exception as e:
        print(f"Yedekleme hatası: {str(e)}")
        return False
    
    # Dosyanın içeriğini oku
    try:
        with open(qt_ui_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Dosya okuma hatası: {str(e)}")
        return False
    
    # İmport hatalarını düzelt
    content = fix_imports(content)
    
    # load_history fonksiyonunu düzelt
    content = fix_load_history(content)
    
    # create_chart_tab fonksiyonunu düzelt
    content = fix_chart_tab(content)
    
    # Diğer sözdizimi hatalarını düzelt
    content = fix_syntax_errors(content)
    
    # Eksik fonksiyonları ekle
    content = add_missing_functions(content)
    
    # Dosyayı yeniden yaz
    try:
        with open(qt_ui_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"qt_ui.py dosyası başarıyla düzeltildi!")
        return True
    except Exception as e:
        print(f"Dosya yazma hatası: {str(e)}")
        return False

def fix_imports(content):
    """Import hatalarını düzeltir"""
    # QPainter ve diğer eksik importları ekle
    if "from PyQt5.QtGui import QPainter" not in content:
        import_section = re.search(r"from PyQt5\.QtGui import.*", content)
        if import_section:
            updated_import = import_section.group(0)
            if "QPainter" not in updated_import:
                updated_import = updated_import.rstrip() + ", QPainter"
                content = content.replace(import_section.group(0), updated_import)
        else:
            # PyQt5.QtGui importu bulunamadıysa, elle ekle
            content = content.replace(
                "from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QSettings",
                "from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QSettings\nfrom PyQt5.QtGui import QColor, QFont, QIcon, QPalette, QPixmap, QPainter"
            )
    
    # QChart import hatalarını düzelt
    try_chart_section = re.search(r"try:\s+from PyQt5\.QtChart import.*?\s+except ImportError:", content, re.DOTALL)
    if try_chart_section:
        updated_section = try_chart_section.group(0)
        if "QValueAxis, QBarCategoryAxis" not in updated_section:
            updated_section = updated_section.replace(
                "from PyQt5.QtChart import QChart, QChartView, QLineSeries, QBarSet, QBarSeries, QPieSeries",
                "from PyQt5.QtChart import QChart, QChartView, QLineSeries, QBarSet, QBarSeries, QPieSeries, QValueAxis, QBarCategoryAxis"
            )
            content = content.replace(try_chart_section.group(0), updated_section)
    
    return content

def fix_load_history(content):
    """load_history fonksiyonunu düzeltir"""
    # load_history fonksiyonunu ara
    load_history_match = re.search(r"def load_history\(self\):.*?try:.*?except Exception as e:", content, re.DOTALL)
    
    if load_history_match:
        # Doğru fonksiyonla değiştir
        corrected_function = """    def load_history(self):
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
        
        except Exception as e:"""
        
        content = content.replace(load_history_match.group(0), corrected_function)
    
    return content

def fix_chart_tab(content):
    """create_chart_tab fonksiyonunu düzeltir"""
    # Hatalı olan kısmı bul
    chart_tab_match = re.search(r"def create_chart_tab\(self, parent_layout\):.*?except ImportError:", content, re.DOTALL)
    
    if chart_tab_match:
        chart_section = chart_tab_match.group(0)
        
        # Eğer hatalar düzelmeyecekse, tamamen değiştir
        if "# Eğer veri yoksa, örnek veriler ek" in chart_section:
            corrected_function = """    def create_chart_tab(self, parent_layout):
        """Grafik sekmesini oluşturur"""
        try:
            from PyQt5.QtChart import QChart, QChartView, QLineSeries, QBarSet, QBarSeries, QPieSeries, QValueAxis, QBarCategoryAxis
            
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
                pie_series.append("Player", 45)
                pie_series.append("Banker", 50)
                pie_series.append("Tie", 5)
            
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
        except ImportError:
            # PyQtChart yoksa, basit bir mesaj göster
            chart_placeholder = QLabel("Grafik gösterimi için PyQtChart modülü gereklidir.\\npip install PyQtChart komutunu çalıştırın.")
            chart_placeholder.setAlignment(Qt.AlignCenter)
            chart_placeholder.setStyleSheet("background-color: #34495E; padding: 80px; border-radius: 4px;")
            parent_layout.addWidget(chart_placeholder)"""
            
            content = content.replace(chart_tab_match.group(0), corrected_function)
    
    return content

def fix_syntax_errors(content):
    """Sözdizimi hatalarını düzeltir"""
    # Eksik parantezleri düzelt
    content = re.sub(r"QTableWidgetItem\([^)]*$", "QTableWidgetItem(\"-\")", content)
    
    # progress_callback hatası düzeltme
    if "def progress_callback" in content and "progress_callback(epoch" not in content:
        progress_callback_match = re.search(r"def progress_callback\(.*?\):.*?pass", content, re.DOTALL)
        if progress_callback_match:
            content = content.replace(
                progress_callback_match.group(0),
                "def progress_callback(epoch, total_epochs, metrics):\n            progress = int(20 + (epoch / total_epochs) * 70)\n            self.progress_update.emit(progress, f\"Epoch {epoch}/{total_epochs}: Doğruluk={metrics.get('accuracy', 0):.4f}\")"
            )
    
    return content

def add_missing_functions(content):
    """Eksik fonksiyonları ekle"""
    # update_progress fonksiyonu eksikse ekle
    if "def update_progress(self, value, message):" not in content:
        class_end = content.rfind("def show_about")
        if class_end != -1:
            insert_position = content.find("\n", content.find("def show_about"))
            missing_function = """
    def update_progress(self, value, message):
        """İlerleme çubuğunu günceller"""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Durum: {message}")
"""
            content = content[:insert_position] + missing_function + content[insert_position:]
    
    return content

def main():
    """Ana fonksiyon"""
    print("PyQt5 arayüzü düzeltme aracı")
    print("----------------------------")
    
    try:
        if fix_qt_ui_file():
            print("\nİşlem tamamlandı! qt_ui.py dosyası başarıyla düzeltildi.")
            print("Uygulamayı şu şekilde çalıştırabilirsiniz: python main.py")
        else:
            print("\nDüzeltme sırasında hatalar oluştu. Lütfen hata mesajlarını kontrol edin.")
    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())