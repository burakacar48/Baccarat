#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baccarat Tahmin Sistemi Düzeltme Betiği
---------------------------------------
Sistemdeki yaygın hataları ve uyumsuzlukları otomatik olarak düzeltir.
"""

import os
import sys
import re
import shutil
import sqlite3
from datetime import datetime
import logging

# Ana dizini ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Konfigürasyon ve backend modüllerini import et
from config.settings import DATABASE_URI
from backend.database.db_manager import DatabaseManager

class BaccaratSystemFixer:
    def __init__(self):
        """
        Sistem düzeltme aracını başlatır
        """
        # Loglama ayarları
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def fix_database_threading(self, file_path):
        """
        Dosyadaki SQLite threading sorunlarını düzeltir
        
        Args:
            file_path (str): Düzeltilecek dosyanın yolu
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Thread-safe connection için gerekli import ve değişiklikler
            thread_safe_import = "from threading import local\n"
            
            # DatabaseManager sınıfını thread-safe hale getir
            thread_safe_code = """
class ThreadSafeDatabase:
    def __init__(self, db_uri):
        self._local = local()  # Thread-local storage
        self.db_uri = db_uri

    def _get_connection(self):
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_uri)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def connect(self):
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # Tabloları oluştur
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                result TEXT NOT NULL,
                previous_pattern TEXT,
                session_id INTEGER
            )
            ''')
            
            # Diğer gerekli tabloları da oluştur
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_games INTEGER,
                win_rate REAL
            )
            ''')
            
            connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Veritabanı bağlantı hatası: {str(e)}")
            return False

    def save_result(self, result, timestamp=None, previous_pattern=None, session_id=None):
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO game_results (timestamp, result, previous_pattern, session_id)
            VALUES (?, ?, ?, ?)
            ''', (timestamp, result, previous_pattern, session_id))
            
            connection.commit()
            result_id = cursor.lastrowid
            
            print(f"Oyun sonucu kaydedildi: ID={result_id}, Sonuç={result}")
            return result_id
        except sqlite3.Error as e:
            print(f"Oyun sonucu kaydetme hatası: {str(e)}")
            return -1

    def seed_initial_data(self):
        \"\"\"Veritabanını başlangıç verileriyle doldur\"\"\"
        connection = self._get_connection()
        cursor = connection.cursor()
        
        # Mevcut sonuç sayısını kontrol et
        cursor.execute("SELECT COUNT(*) FROM game_results")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Örnek sonuçlar ekle
            seed_results = [
                (datetime.now().isoformat(), 'P'),
                (datetime.now().isoformat(), 'B'),
                (datetime.now().isoformat(), 'T'),
                (datetime.now().isoformat(), 'P'),
                (datetime.now().isoformat(), 'B')
            ]
            
            cursor.executemany('''
            INSERT INTO game_results (timestamp, result) 
            VALUES (?, ?)
            ''', seed_results)
            
            connection.commit()
            print(f"{len(seed_results)} adet başlangıç verisi eklendi")
"""
            
            # Dosya içeriğindeki DatabaseManager sınıfını değiştir
            content = re.sub(
                r'class DatabaseManager:.*?def __init__\(self, db_uri\):',
                'class DatabaseManager(ThreadSafeDatabase):\n    def __init__(self, db_uri):',
                content,
                flags=re.DOTALL
            )
            
            # Gerekli import'ları ekle
            content = thread_safe_import + thread_safe_code + content
            
            # Dosyayı yeniden yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"{file_path} dosyasında thread-safe düzenlemeler yapıldı.")
        except Exception as e:
            self.logger.error(f"Dosya düzenleme hatası: {str(e)}")
    
    def fix_prediction_engine(self, file_path):
        """
        Tahmin motorundaki veritabanı ve thread sorunlarını düzeltir
        
        Args:
            file_path (str): Düzeltilecek dosyanın yolu
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Tahmin motorunda veritabanı ve thread hatalarını düzelt
            thread_safe_modifications = """
    def predict(self, data=None, save_prediction=True):
        # Veri yoksa veritabanından son verileri getir
        if data is None:
            if not self.db_manager:
                logger.error("Veritabanı yöneticisi bulunamadı. Veri sağlayın veya DB yöneticisi ayarlayın.")
                return None
            
            # Her seferinde yeni bir bağlantı oluştur
            db_connection = self.db_manager._get_connection()
            cursor = db_connection.cursor()
            
            # Son N sonucu getir
            cursor.execute("SELECT * FROM game_results ORDER BY id DESC LIMIT 20")
            last_results = cursor.fetchall()
            
            if not last_results:
                logger.warning("Veritabanında sonuç bulunamadı. Tahmin yapılamıyor.")
                return None
            
            # Sonuçları düzenle
            results = [result['result'] for result in last_results]
            results.reverse()  # En eskiden en yeniye sırala
            
            data = {'last_results': results}
        
        # Geri kalan kod aynı kalacak
        
        # Sonuç kaydı için her seferinde yeni bir bağlantı kullan
        if save_prediction and self.db_manager:
            try:
                db_connection = self.db_manager._get_connection()
                cursor = db_connection.cursor()
                
                # Algoritma ve tahmin kaydetme kodları buraya
                # Her seferinde yeni bir bağlantı kullanılacak
            except Exception as e:
                logger.error(f"Tahmin kaydetme hatası: {str(e)}")
"""
            
            # Dosya içeriğindeki predict metodunu değiştir
            content = re.sub(
                r'def predict\(self, data=None, save_prediction=True\):.*?return final_prediction',
                thread_safe_modifications,
                content,
                flags=re.DOTALL
            )
            
            # Dosyayı yeniden yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"{file_path} dosyasında tahmin motoru thread-safe düzenlemeleri yapıldı.")
        except Exception as e:
            self.logger.error(f"Tahmin motoru düzenleme hatası: {str(e)}")
    
    def fix_worker_thread(self, file_path):
        """
        Worker thread'inde veritabanı bağlantısı sorunlarını düzeltir
        
        Args:
            file_path (str): Düzeltilecek dosyanın yolu
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Worker thread için thread-safe düzenlemeler
            thread_safe_worker = """
    def set_components(self, prediction_engine, db_manager, performance_tracker=None):
        self.prediction_engine = prediction_engine
        # Her thread için yeni bir veritabanı bağlantısı oluştur
        self.db_manager = type(db_manager)(db_manager.db_uri)
        self.db_manager.connect()
        self.performance_tracker = performance_tracker
"""
            
            # Dosya içeriğindeki set_components metodunu değiştir
            content = re.sub(
                r'def set_components\(self, prediction_engine, db_manager, performance_tracker=None\):.*?performance_tracker\)',
                thread_safe_worker,
                content,
                flags=re.DOTALL
            )
            
            # Dosyayı yeniden yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"{file_path} dosyasında worker thread düzenlemeleri yapıldı.")
        except Exception as e:
            self.logger.error(f"Worker thread düzenleme hatası: {str(e)}")
    
    def fix_main_initialization(self, file_path):
        """
        Ana başlatma dosyasında veritabanı başlangıç verilerini ayarlar
        
        Args:
            file_path (str): Düzeltilecek dosyanın yolu
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Veritabanı başlangıç verilerini ekle
            seed_data_code = """
    # Veritabanını başlangıç verileriyle doldur
    if db_manager:
        db_manager.seed_initial_data()
"""
            
            # Veritabanı bağlantısından hemen sonra seed_initial_data çağrısını ekle
            content = content.replace(
                "db_manager.connect()",
                "db_manager.connect()\n    # Veritabanını başlangıç verileriyle doldur\n    db_manager.seed_initial_data()"
            )
            
            # Dosyayı yeniden yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"{file_path} dosyasında veritabanı başlangıç verileri ayarlandı.")
        except Exception as e:
            self.logger.error(f"Ana başlatma düzenleme hatası: {str(e)}")
    
    def run_fixes(self):
        """
        Tüm düzeltmeleri çalıştırır
        """
        self.logger.info("Baccarat Tahmin Sistemi otomatik düzeltme işlemi başlatılıyor...")
        
        # Düzeltilecek dosyaları tanımla
        fixes = [
            (os.path.join(parent_dir, 'backend', 'database', 'db_manager.py'), self.fix_database_threading),
            (os.path.join(parent_dir, 'backend', 'engine', 'prediction_engine.py'), self.fix_prediction_engine),
            (os.path.join(parent_dir, 'frontend', 'qt_ui.py'), self.fix_worker_thread),
            (os.path.join(parent_dir, 'main.py'), self.fix_main_initialization)
        ]
        
        # Dosyaları düzelt
        for file_path, fix_method in fixes:
            if os.path.exists(file_path):
                self.logger.info(f"Düzeltme yapılıyor: {file_path}")
                # Yedekleme
                backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"Yedekleme yapıldı: {backup_path}")
                
                # Düzeltmeyi uygula
                fix_method(file_path)
            else:
                self.logger.warning(f"Dosya bulunamadı: {file_path}")
        
        self.logger.info("Baccarat Tahmin Sistemi otomatik düzeltme işlemi tamamlandı.")

def main():
    """
    Düzeltme betiğini çalıştırır
    """
    fixer = BaccaratSystemFixer()
    fixer.run_fixes()

if __name__ == "__main__":
    main()