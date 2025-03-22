#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veritabanı Yöneticisi
---------------------
Veritabanı işlemlerini yönetir, sonuçları kaydeder ve sorgular.
"""

import os
import logging
import sqlite3
import threading
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Veritabanı işlemlerini yöneten thread-safe sınıf
    
    SQLite veritabanını kullanarak oyun sonuçlarını, tahminleri ve
    algoritma performanslarını depolar ve yönetir.
    """
    
    def __init__(self, db_uri):
        """
        DatabaseManager sınıfını başlatır
        
        Args:
            db_uri (str): Veritabanı bağlantı URI'si
        """
        self.db_uri = db_uri
        self._local = threading.local()  # Thread-local storage
        
        # Veritabanı klasörünün varlığını kontrol et
        db_dir = os.path.dirname(db_uri)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Veritabanı dizini oluşturuldu: {db_dir}")
    
    def _get_connection(self):
        """
        Thread-local veritabanı bağlantısını sağlar
        
        Returns:
            sqlite3.Connection: Veritabanı bağlantısı
        """
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_uri)
            self._local.connection.row_factory = sqlite3.Row  # Sütun adlarıyla erişim için
        return self._local.connection
    
    def connect(self):
        """
        Veritabanına bağlantı kurar ve gerekli tabloları oluşturur
        
        Returns:
            bool: Bağlantı başarılı ise True, değilse False
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # Tabloları oluştur
            self._create_tables(cursor)
                
            connection.commit()
            logger.info(f"Veritabanı bağlantısı kuruldu: {self.db_uri}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Veritabanı bağlantı hatası: {str(e)}")
            return False
    
    def _create_tables(self, cursor):
        """
        Gerekli veritabanı tablolarını oluşturur
        
        Args:
            cursor (sqlite3.Cursor): Veritabanı işaretçisi
        """
        # GAME_RESULTS tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            result TEXT NOT NULL,
            previous_pattern TEXT,
            session_id INTEGER
        )
        ''')
        
        # PREDICTIONS tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL,
            game_result_id INTEGER,
            FOREIGN KEY (game_result_id) REFERENCES game_results(id)
        )
        ''')

        # ALGORITHM_PERFORMANCE tablosu     
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithm_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL
        )
        ''')
        
        # SESSIONS tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
    
    def disconnect(self):
        """
        Veritabanı bağlantısını kapatır
        """
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection
            logger.info("Veritabanı bağlantısı kapatıldı")
    
    def seed_initial_data(self):
        """Veritabanını başlangıç verileriyle doldur"""
        try:
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
                logger.info(f"{len(seed_results)} adet başlangıç verisi eklendi")
            else:
                logger.info(f"Veritabanında zaten {count} adet kayıt mevcut")
        except sqlite3.Error as e:
            logger.error(f"Başlangıç verisi ekleme hatası: {str(e)}")

    def save_result(self, result, previous_pattern=None, session_id=None):
        """
        Oyun sonucunu veritabanına kaydeder

        Args:
            result (str): Oyun sonucu (P, T veya B)
            previous_pattern (str, optional): Önceki desen. Varsayılan None.
            session_id (int, optional): Oturum ID'si. Varsayılan None.
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()

            cursor.execute('''
            INSERT INTO game_results (timestamp, result, previous_pattern, session_id)
            VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), result, previous_pattern, session_id))

            connection.commit()
            logger.info(f"Oyun sonucu kaydedildi: {result}")
        except sqlite3.Error as e:
            logger.error(f"Sonuç kaydetme hatası: {str(e)}")

    def get_last_n_results(self, n):
        """
        Son n adet oyun sonucunu döndürür
        
        Args:
            n (int): İstenilen sonuç sayısı

        Returns:
            list: Sonuç listesi (en yeni sonuç başta)
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()

            cursor.execute('''
            SELECT result 
            FROM game_results
            ORDER BY id DESC
            LIMIT ?
            ''', (n,))

            results = [row['result'] for row in cursor.fetchall()]
            logger.info(f"Son {n} sonuç alındı: {results}")
            return results
        except sqlite3.Error as e:
            logger.error(f"Sonuç alma hatası: {str(e)}")
            return []

    def get_all_results(self):
        """
        Tüm oyun sonuçlarını döndürür

        Returns:
            list: Sonuç listesi (en eski sonuç başta)
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()

            cursor.execute('''
            SELECT result
            FROM game_results
            ORDER BY id
            ''')

            results = [row['result'] for row in cursor.fetchall()]
            logger.info(f"Tüm sonuçlar alındı: {len(results)} adet")
            return results
        except sqlite3.Error as e:
            logger.error(f"Sonuç alma hatası: {str(e)}")
            return []
        
    def save_prediction(self, algorithm, prediction, probability=None, game_result_id=None):
        """
        Algoritma tahminini veritabanına kaydeder

        Args:
            algorithm (str): Algoritma adı
            prediction (str): Tahmin (P, T veya B)  
            probability (float, optional): Tahmin olasılığı. Varsayılan None.
            game_result_id (int, optional): İlgili oyun sonucu ID'si. Varsayılan None.
        """
        try:
            connection = self._get_connection() 
            cursor = connection.cursor()

            cursor.execute('''
            INSERT INTO predictions (timestamp, algorithm, prediction, probability, game_result_id)
            VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), algorithm, prediction, probability, game_result_id))

            connection.commit()
            logger.info(f"Algoritma tahmini kaydedildi: {algorithm} - {prediction}")
        except sqlite3.Error as e:
            logger.error(f"Tahmin kaydetme hatası: {str(e)}")

    def save_algorithm_performance(self, algorithm, accuracy, precision, recall, f1_score):
        """
        Algoritma performansını veritabanına kaydeder

        Args:    
            algorithm (str): Algoritma adı
            accuracy (float): Doğruluk oranı
            precision (float): Kesinlik (precision) 
            recall (float): Duyarlılık (recall)
            f1_score (float): F1 skoru
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            cursor.execute('''
            INSERT INTO algorithm_performance (timestamp, algorithm, accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?, ?)  
            ''', (datetime.now().isoformat(), algorithm, accuracy, precision, recall, f1_score))

            connection.commit() 
            logger.info(f"Algoritma performansı kaydedildi: {algorithm}")
        except sqlite3.Error as e:
            logger.error(f"Performans kaydetme hatası: {str(e)}")
            
    def create_session(self):
        """
        Yeni bir oturum başlatır ve oturum ID'sini döndürür
        
        Returns:
            int: Oluşturulan oturumun ID'si
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # Yeni oturum kaydı oluştur
            cursor.execute("INSERT INTO sessions DEFAULT VALUES")
            session_id = cursor.lastrowid
            
            connection.commit()
            logger.info(f"Yeni oturum oluşturuldu: ID={session_id}") 
            return session_id
        except sqlite3.Error as e:
            logger.error(f"Oturum oluşturma hatası: {str(e)}")
            return None