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
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Veritabanı işlemlerini yöneten sınıf
    
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
        self.connection = None
        self.cursor = None
        
        # Veritabanı klasörünün varlığını kontrol et
        db_dir = os.path.dirname(db_uri)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Veritabanı dizini oluşturuldu: {db_dir}")
    
    def connect(self):
        """
        Veritabanına bağlantı kurar ve gerekli tabloları oluşturur
        
        Returns:
            bool: Bağlantı başarılı ise True, değilse False
        """
        try:
            self.connection = sqlite3.connect(self.db_uri)
            self.connection.row_factory = sqlite3.Row  # Sütun adlarıyla erişim için
            self.cursor = self.connection.cursor()
            
            # Tabloları oluştur
            self._create_tables()
            
            logger.info(f"Veritabanı bağlantısı kuruldu: {self.db_uri}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Veritabanı bağlantı hatası: {str(e)}")
            return False
    
    def disconnect(self):
        """
        Veritabanı bağlantısını kapatır
        """
        if self.connection:
            self.connection.close()
            logger.info("Veritabanı bağlantısı kapatıldı")
    
    def _create_tables(self):
        """
        Gerekli veritabanı tablolarını oluşturur
        """
        # GAME_RESULTS tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            result TEXT NOT NULL,
            previous_pattern TEXT,
            session_id INTEGER
        )
        ''')
        
        # PREDICTIONS tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_result_id INTEGER,
            algorithm_id INTEGER,
            predicted_result TEXT NOT NULL,
            is_correct INTEGER,
            confidence_score REAL,
            FOREIGN KEY (game_result_id) REFERENCES game_results (id),
            FOREIGN KEY (algorithm_id) REFERENCES algorithms (id)
        )
        ''')
        
        # ALGORITHMS tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT,
            current_accuracy REAL,
            weight REAL,
            last_updated TEXT
        )
        ''')
        
        # PATTERNS tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_sequence TEXT NOT NULL,
            occurrence_count INTEGER,
            next_result TEXT,
            success_rate REAL
        )
        ''')
        
        # MODEL_VERSIONS tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            file_path TEXT,
            accuracy REAL,
            is_active INTEGER
        )
        ''')
        
        # SESSIONS tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_games INTEGER,
            win_rate REAL
        )
        ''')
        
        # ALGORITHM_PERFORMANCE tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithm_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm_id INTEGER,
            evaluation_date TEXT NOT NULL,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL,
            FOREIGN KEY (algorithm_id) REFERENCES algorithms (id)
        )
        ''')
        
        # DEEP_LEARNING_FEATURES tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS deep_learning_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_result_id INTEGER,
            feature_vector TEXT,
            created_at TEXT NOT NULL,
            model_version_id INTEGER,
            FOREIGN KEY (game_result_id) REFERENCES game_results (id),
            FOREIGN KEY (model_version_id) REFERENCES model_versions (id)
        )
        ''')
        
        self.connection.commit()
        logger.debug("Veritabanı tabloları oluşturuldu")
    
    def reset_database(self):
        """
        Veritabanını sıfırlar (tüm tabloları siler ve yeniden oluşturur)
        """
        try:
            if not self.connection:
                self.connect()
            
            # Var olan tabloları sil
            tables = [
                "game_results", "predictions", "algorithms", "patterns",
                "model_versions", "sessions", "algorithm_performance", "deep_learning_features"
            ]
            
            for table in tables:
                self.cursor.execute(f"DROP TABLE IF EXISTS {table}")
            
            # Tabloları yeniden oluştur
            self._create_tables()
            
            logger.warning("Veritabanı sıfırlandı, tüm tablolar silindi ve yeniden oluşturuldu")
            return True
        except sqlite3.Error as e:
            logger.error(f"Veritabanı sıfırlama hatası: {str(e)}")
            return False
    
    def save_result(self, result, timestamp=None, previous_pattern=None, session_id=None):
        """
        Oyun sonucunu veritabanına kaydeder
        
        Args:
            result (str): Oyun sonucu (P/B/T)
            timestamp (str, optional): Zaman damgası, None ise şu anki zaman kullanılır
            previous_pattern (str, optional): Önceki desen
            session_id (int, optional): Oturum ID
        
        Returns:
            int: Eklenen kaydın ID'si, hata durumunda -1
        """
        try:
            if not self.connection:
                self.connect()
            
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            self.cursor.execute('''
            INSERT INTO game_results (timestamp, result, previous_pattern, session_id)
            VALUES (?, ?, ?, ?)
            ''', (timestamp, result, previous_pattern, session_id))
            
            self.connection.commit()
            result_id = self.cursor.lastrowid
            
            logger.debug(f"Oyun sonucu kaydedildi: ID={result_id}, Sonuç={result}")
            return result_id
        except sqlite3.Error as e:
            logger.error(f"Oyun sonucu kaydetme hatası: {str(e)}")
            return -1
    
    def save_prediction(self, game_result_id, algorithm_id, predicted_result, is_correct=None, confidence_score=0.0):
        """
        Tahmin sonucunu veritabanına kaydeder
        
        Args:
            game_result_id (int): İlgili oyun sonucunun ID'si
            algorithm_id (int): Algoritmanın ID'si
            predicted_result (str): Tahmin edilen sonuç (P/B/T)
            is_correct (bool, optional): Tahminin doğru olup olmadığı
            confidence_score (float, optional): Güven skoru
        
        Returns:
            int: Eklenen kaydın ID'si, hata durumunda -1
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            INSERT INTO predictions (game_result_id, algorithm_id, predicted_result, is_correct, confidence_score)
            VALUES (?, ?, ?, ?, ?)
            ''', (game_result_id, algorithm_id, predicted_result, 1 if is_correct else 0, confidence_score))
            
            self.connection.commit()
            prediction_id = self.cursor.lastrowid
            
            logger.debug(f"Tahmin kaydedildi: ID={prediction_id}, Algoritma={algorithm_id}, Sonuç={predicted_result}")
            return prediction_id
        except sqlite3.Error as e:
            logger.error(f"Tahmin kaydetme hatası: {str(e)}")
            return -1
    
    def save_algorithm(self, name, algorithm_type, accuracy=0.0, weight=1.0):
        """
        Algoritma bilgilerini veritabanına kaydeder
        
        Args:
            name (str): Algoritmanın adı
            algorithm_type (str): Algoritma tipi
            accuracy (float, optional): Algoritmanın doğruluk oranı
            weight (float, optional): Algoritmanın ağırlığı
        
        Returns:
            int: Eklenen kaydın ID'si, hata durumunda -1
        """
        try:
            if not self.connection:
                self.connect()
            
            # Algoritma zaten var mı kontrol et
            self.cursor.execute('''
            SELECT id FROM algorithms WHERE name = ?
            ''', (name,))
            
            row = self.cursor.fetchone()
            if row:
                algorithm_id = row['id']
                
                # Algoritmayı güncelle
                self.cursor.execute('''
                UPDATE algorithms 
                SET current_accuracy = ?, weight = ?, last_updated = ?
                WHERE id = ?
                ''', (accuracy, weight, datetime.now().isoformat(), algorithm_id))
                
                self.connection.commit()
                logger.debug(f"Algoritma güncellendi: ID={algorithm_id}, Ad={name}, Doğruluk={accuracy}")
                return algorithm_id
            
            # Yeni algoritma ekle
            self.cursor.execute('''
            INSERT INTO algorithms (name, type, current_accuracy, weight, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ''', (name, algorithm_type, accuracy, weight, datetime.now().isoformat()))
            
            self.connection.commit()
            algorithm_id = self.cursor.lastrowid
            
            logger.debug(f"Algoritma kaydedildi: ID={algorithm_id}, Ad={name}")
            return algorithm_id
        except sqlite3.Error as e:
            logger.error(f"Algoritma kaydetme hatası: {str(e)}")
            return -1
    
    def update_algorithm_performance(self, algorithm_id, total_predictions, correct_predictions, accuracy):
        """
        Algoritma performans bilgilerini günceller
        
        Args:
            algorithm_id (int): Algoritmanın ID'si
            total_predictions (int): Toplam tahmin sayısı
            correct_predictions (int): Doğru tahmin sayısı
            accuracy (float): Doğruluk oranı
        
        Returns:
            bool: İşlem başarılı ise True, değilse False
        """
        try:
            if not self.connection:
                self.connect()
            
            # Algoritma performans kaydı ekle
            self.cursor.execute('''
            INSERT INTO algorithm_performance (algorithm_id, evaluation_date, total_predictions, correct_predictions, accuracy)
            VALUES (?, ?, ?, ?, ?)
            ''', (algorithm_id, datetime.now().isoformat(), total_predictions, correct_predictions, accuracy))
            
            # Algoritma bilgilerini güncelle
            self.cursor.execute('''
            UPDATE algorithms 
            SET current_accuracy = ?, last_updated = ?
            WHERE id = ?
            ''', (accuracy, datetime.now().isoformat(), algorithm_id))
            
            self.connection.commit()
            logger.debug(f"Algoritma performansı güncellendi: ID={algorithm_id}, Doğruluk={accuracy:.4f}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Algoritma performansı güncelleme hatası: {str(e)}")
            return False
    
    def get_last_n_results(self, n=10):
        """
        Son N oyun sonucunu getirir
        
        Args:
            n (int): Getirilecek sonuç sayısı
        
        Returns:
            list: Sonuçların listesi
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            SELECT * FROM game_results 
            ORDER BY id DESC 
            LIMIT ?
            ''', (n,))
            
            rows = self.cursor.fetchall()
            
            # Sonuçları dönüştür
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'result': row['result'],
                    'previous_pattern': row['previous_pattern'],
                    'session_id': row['session_id']
                })
            
            logger.debug(f"Son {len(results)} sonuç getirildi")
            return results
        except sqlite3.Error as e:
            logger.error(f"Sonuç getirme hatası: {str(e)}")
            return []
    
    def get_all_results(self):
        """
        Tüm oyun sonuçlarını getirir
        
        Returns:
            list: Tüm sonuçların listesi (sadece P/B/T karakterleri)
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            SELECT result FROM game_results 
            ORDER BY id ASC
            ''')
            
            rows = self.cursor.fetchall()
            
            # Sadece sonuçları dönüştür
            results = [row['result'] for row in rows]
            
            logger.debug(f"Toplam {len(results)} sonuç getirildi")
            return results
        except sqlite3.Error as e:
            logger.error(f"Sonuç getirme hatası: {str(e)}")
            return []
    
    def get_algorithm_by_name(self, name):
        """
        İsme göre algoritma bilgilerini getirir
        
        Args:
            name (str): Algoritmanın adı
        
        Returns:
            dict: Algoritma bilgileri, bulunamazsa None
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            SELECT * FROM algorithms WHERE name = ?
            ''', (name,))
            
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'type': row['type'],
                    'current_accuracy': row['current_accuracy'],
                    'weight': row['weight'],
                    'last_updated': row['last_updated']
                }
            
            return None
        except sqlite3.Error as e:
            logger.error(f"Algoritma getirme hatası: {str(e)}")
            return None
    
    def save_pattern(self, pattern_sequence, next_result, increment=True):
        """
        Desen bilgisini kaydeder veya günceller
        
        Args:
            pattern_sequence (str): Desen dizisi
            next_result (str): Desenden sonraki sonuç
            increment (bool): Sayacı artır
        
        Returns:
            int: Eklenen/güncellenen kaydın ID'si, hata durumunda -1
        """
        try:
            if not self.connection:
                self.connect()
            
            # Desen zaten var mı kontrol et
            self.cursor.execute('''
            SELECT id, occurrence_count, next_result, success_rate 
            FROM patterns 
            WHERE pattern_sequence = ? AND next_result = ?
            ''', (pattern_sequence, next_result))
            
            row = self.cursor.fetchone()
            
            if row:
                pattern_id = row['id']
                occurrence_count = row['occurrence_count']
                
                if increment:
                    occurrence_count += 1
                
                # Deseni güncelle
                self.cursor.execute('''
                UPDATE patterns 
                SET occurrence_count = ?
                WHERE id = ?
                ''', (occurrence_count, pattern_id))
                
                self.connection.commit()
                return pattern_id
            
            # Yeni desen ekle
            self.cursor.execute('''
            INSERT INTO patterns (pattern_sequence, occurrence_count, next_result, success_rate)
            VALUES (?, ?, ?, ?)
            ''', (pattern_sequence, 1, next_result, 0.0))
            
            self.connection.commit()
            pattern_id = self.cursor.lastrowid
            
            return pattern_id
        except sqlite3.Error as e:
            logger.error(f"Desen kaydetme hatası: {str(e)}")
            return -1
    
    def get_pattern_stats(self, pattern_sequence):
        """
        Belirli bir desenin istatistiklerini getirir
        
        Args:
            pattern_sequence (str): Desen dizisi
        
        Returns:
            dict: Desen istatistikleri
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            SELECT next_result, occurrence_count 
            FROM patterns 
            WHERE pattern_sequence = ?
            ORDER BY occurrence_count DESC
            ''', (pattern_sequence,))
            
            rows = self.cursor.fetchall()
            
            stats = {
                'pattern': pattern_sequence,
                'total_occurrences': 0,
                'next_results': {}
            }
            
            for row in rows:
                next_result = row['next_result']
                count = row['occurrence_count']
                stats['next_results'][next_result] = count
                stats['total_occurrences'] += count
            
            # Her sonucun olasılığını hesapla
            if stats['total_occurrences'] > 0:
                for result, count in stats['next_results'].items():
                    stats['next_results'][result] = {
                        'count': count,
                        'probability': count / stats['total_occurrences']
                    }
            
            return stats
        except sqlite3.Error as e:
            logger.error(f"Desen istatistikleri getirme hatası: {str(e)}")
            return {
                'pattern': pattern_sequence,
                'total_occurrences': 0,
                'next_results': {}
            }
    
    def save_model_version(self, model_type, file_path, accuracy=0.0, is_active=True):
        """
        Model versiyonunu kaydeder
        
        Args:
            model_type (str): Model tipi
            file_path (str): Model dosya yolu
            accuracy (float): Model doğruluk oranı
            is_active (bool): Model aktif mi
        
        Returns:
            int: Eklenen kaydın ID'si, hata durumunda -1
        """
        try:
            if not self.connection:
                self.connect()
            
            # Eğer aktif model kaydediliyorsa, diğer aktif modelleri pasif yap
            if is_active:
                self.cursor.execute('''
                UPDATE model_versions 
                SET is_active = 0
                WHERE model_type = ?
                ''', (model_type,))
            
            # Yeni model versiyonu ekle
            self.cursor.execute('''
            INSERT INTO model_versions (model_type, created_at, file_path, accuracy, is_active)
            VALUES (?, ?, ?, ?, ?)
            ''', (model_type, datetime.now().isoformat(), file_path, accuracy, 1 if is_active else 0))
            
            self.connection.commit()
            model_id = self.cursor.lastrowid
            
            logger.info(f"Model versiyonu kaydedildi: ID={model_id}, Tip={model_type}, Dosya={file_path}")
            return model_id
        except sqlite3.Error as e:
            logger.error(f"Model versiyonu kaydetme hatası: {str(e)}")
            return -1
    
    def get_active_model(self, model_type):
        """
        Aktif model versiyonunu getirir
        
        Args:
            model_type (str): Model tipi
        
        Returns:
            dict: Model bilgileri, bulunamazsa None
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            SELECT * FROM model_versions 
            WHERE model_type = ? AND is_active = 1
            ORDER BY id DESC
            LIMIT 1
            ''', (model_type,))
            
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'model_type': row['model_type'],
                    'created_at': row['created_at'],
                    'file_path': row['file_path'],
                    'accuracy': row['accuracy']
                }
            
            return None
        except sqlite3.Error as e:
            logger.error(f"Aktif model getirme hatası: {str(e)}")
            return None
    
    def create_session(self):
        """
        Yeni bir oturum oluşturur
        
        Returns:
            int: Oluşturulan oturumun ID'si, hata durumunda -1
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            INSERT INTO sessions (start_time, total_games)
            VALUES (?, ?)
            ''', (datetime.now().isoformat(), 0))
            
            self.connection.commit()
            session_id = self.cursor.lastrowid
            
            logger.info(f"Yeni oturum oluşturuldu: ID={session_id}")
            return session_id
        except sqlite3.Error as e:
            logger.error(f"Oturum oluşturma hatası: {str(e)}")
            return -1
    
    def end_session(self, session_id, total_games, win_rate=0.0):
        """
        Oturumu sonlandırır
        
        Args:
            session_id (int): Oturum ID'si
            total_games (int): Toplam oyun sayısı
            win_rate (float): Kazanç oranı
        
        Returns:
            bool: İşlem başarılı ise True, değilse False
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, total_games = ?, win_rate = ?
            WHERE id = ?
            ''', (datetime.now().isoformat(), total_games, win_rate, session_id))
            
            self.connection.commit()
            logger.info(f"Oturum sonlandırıldı: ID={session_id}, Oyun={total_games}, Oran={win_rate:.4f}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Oturum sonlandırma hatası: {str(e)}")
            return False
    
    def save_deep_learning_features(self, game_result_id, feature_vector, model_version_id=None):
        """
        Derin öğrenme için özellik vektörünü kaydeder
        
        Args:
            game_result_id (int): İlgili oyun sonucunun ID'si
            feature_vector (dict/list): Özellik vektörü
            model_version_id (int, optional): Model versiyonu ID'si
        
        Returns:
            int: Eklenen kaydın ID'si, hata durumunda -1
        """
        try:
            if not self.connection:
                self.connect()
            
            # Özellik vektörünü JSON formatına dönüştür
            feature_json = json.dumps(feature_vector)
            
            self.cursor.execute('''
            INSERT INTO deep_learning_features (game_result_id, feature_vector, created_at, model_version_id)
            VALUES (?, ?, ?, ?)
            ''', (game_result_id, feature_json, datetime.now().isoformat(), model_version_id))
            
            self.connection.commit()
            feature_id = self.cursor.lastrowid
            
            logger.debug(f"Derin öğrenme özellikleri kaydedildi: ID={feature_id}, Oyun={game_result_id}")
            return feature_id
        except sqlite3.Error as e:
            logger.error(f"Derin öğrenme özellikleri kaydetme hatası: {str(e)}")
            return -1
    
    def get_features_for_training(self, limit=1000):
        """
        Eğitim için derin öğrenme özelliklerini getirir
        
        Args:
            limit (int): Maksimum kayıt sayısı
        
        Returns:
            list: Özellik vektörleri ve sonuçları içeren liste
        """
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute('''
            SELECT df.feature_vector, gr.result 
            FROM deep_learning_features df
            JOIN game_results gr ON df.game_result_id = gr.id
            ORDER BY df.id DESC 
            LIMIT ?
            ''', (limit,))
            
            rows = self.cursor.fetchall()
            
            features = []
            for row in rows:
                feature_vector = json.loads(row['feature_vector'])
                result = row['result']
                features.append({
                    'features': feature_vector,
                    'result': result
                })
            
            logger.debug(f"Eğitim için {len(features)} özellik vektörü getirildi")
            return features
        except sqlite3.Error as e:
            logger.error(f"Eğitim özellikleri getirme hatası: {str(e)}")
            return []