#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veritabanı Şeması
---------------
SQLite tablo oluşturma SQL ifadeleri.
"""

# Tablo oluşturma sorguları
CREATE_TABLES_QUERIES = {
    'game_results': """
    CREATE TABLE IF NOT EXISTS game_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        result TEXT NOT NULL, -- P/B/T
        previous_pattern TEXT,
        session_id INTEGER,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    )
    """,
    
    'predictions': """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_result_id INTEGER,
        algorithm_id INTEGER,
        predicted_result TEXT NOT NULL, -- P/B/T
        is_correct INTEGER,
        confidence_score REAL,
        FOREIGN KEY (game_result_id) REFERENCES game_results (id),
        FOREIGN KEY (algorithm_id) REFERENCES algorithms (id)
    )
    """,
    
    'algorithms': """
    CREATE TABLE IF NOT EXISTS algorithms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT,
        current_accuracy REAL,
        weight REAL,
        last_updated TEXT
    )
    """,
    
    'patterns': """
    CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_sequence TEXT NOT NULL,
        occurrence_count INTEGER,
        next_result TEXT,
        success_rate REAL
    )
    """,
    
    'model_versions': """
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        file_path TEXT,
        accuracy REAL,
        is_active INTEGER
    )
    """,
    
    'sessions': """
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT NOT NULL,
        end_time TEXT,
        total_games INTEGER,
        win_rate REAL
    )
    """,
    
    'algorithm_performance': """
    CREATE TABLE IF NOT EXISTS algorithm_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        algorithm_id INTEGER,
        evaluation_date TEXT NOT NULL,
        total_predictions INTEGER,
        correct_predictions INTEGER,
        accuracy REAL,
        FOREIGN KEY (algorithm_id) REFERENCES algorithms (id)
    )
    """,
    
    'deep_learning_features': """
    CREATE TABLE IF NOT EXISTS deep_learning_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_result_id INTEGER,
        feature_vector TEXT, -- JSON formatında
        created_at TEXT NOT NULL,
        model_version_id INTEGER,
        FOREIGN KEY (game_result_id) REFERENCES game_results (id),
        FOREIGN KEY (model_version_id) REFERENCES model_versions (id)
    )
    """
}

# Örnek veriler
SAMPLE_DATA_QUERIES = {
    'algorithms': """
    INSERT INTO algorithms (name, type, current_accuracy, weight, last_updated)
    VALUES 
        ('Pattern Analysis', 'pattern', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Statistical Model', 'statistical', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Sequence Analysis', 'pattern', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Bayes Model', 'statistical', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Markov Chain', 'markov', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Monte Carlo Simulation', 'simulation', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Cyclical Analysis', 'pattern', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Correlation Model', 'statistical', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Clustering Model', 'machine_learning', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Entropy Model', 'statistical', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Regression Model', 'machine_learning', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Time Series Model', 'machine_learning', 0.0, 1.0, CURRENT_TIMESTAMP),
        ('Combination Analysis', 'ensemble', 0.0, 1.5, CURRENT_TIMESTAMP),
        ('LSTM', 'deep_learning', 0.0, 1.5, CURRENT_TIMESTAMP)
    """
}

# İndeksler
CREATE_INDEXES_QUERIES = {
    'idx_game_results_timestamp': """
    CREATE INDEX IF NOT EXISTS idx_game_results_timestamp ON game_results (timestamp)
    """,
    
    'idx_predictions_game_result_id': """
    CREATE INDEX IF NOT EXISTS idx_predictions_game_result_id ON predictions (game_result_id)
    """,
    
    'idx_predictions_algorithm_id': """
    CREATE INDEX IF NOT EXISTS idx_predictions_algorithm_id ON predictions (algorithm_id)
    """,
    
    'idx_patterns_sequence': """
    CREATE INDEX IF NOT EXISTS idx_patterns_sequence ON patterns (pattern_sequence)
    """,
    
    'idx_algorithm_performance_algorithm_id': """
    CREATE INDEX IF NOT EXISTS idx_algorithm_performance_algorithm_id ON algorithm_performance (algorithm_id)
    """,
    
    'idx_deep_learning_features_game_result_id': """
    CREATE INDEX IF NOT EXISTS idx_deep_learning_features_game_result_id ON deep_learning_features (game_result_id)
    """
}

# Görünümler (Views)
CREATE_VIEWS_QUERIES = {
    'view_algorithm_stats': """
    CREATE VIEW IF NOT EXISTS view_algorithm_stats AS
    SELECT 
        a.id AS algorithm_id,
        a.name AS algorithm_name,
        a.type AS algorithm_type,
        a.weight AS weight,
        COUNT(p.id) AS total_predictions,
        SUM(p.is_correct) AS correct_predictions,
        CASE WHEN COUNT(p.id) > 0 THEN CAST(SUM(p.is_correct) AS REAL) / COUNT(p.id) ELSE 0 END AS accuracy
    FROM algorithms a
    LEFT JOIN predictions p ON a.id = p.algorithm_id
    GROUP BY a.id
    """,
    
    'view_recent_performance': """
    CREATE VIEW IF NOT EXISTS view_recent_performance AS
    SELECT 
        a.id AS algorithm_id,
        a.name AS algorithm_name,
        COUNT(p.id) AS total_predictions,
        SUM(p.is_correct) AS correct_predictions,
        CASE WHEN COUNT(p.id) > 0 THEN CAST(SUM(p.is_correct) AS REAL) / COUNT(p.id) ELSE 0 END AS accuracy,
        MAX(gr.timestamp) AS last_prediction_time
    FROM algorithms a
    JOIN predictions p ON a.id = p.algorithm_id
    JOIN game_results gr ON p.game_result_id = gr.id
    WHERE gr.timestamp >= datetime('now', '-30 day')
    GROUP BY a.id
    """,
    
    'view_pattern_statistics': """
    CREATE VIEW IF NOT EXISTS view_pattern_statistics AS
    SELECT 
        pattern_sequence,
        next_result,
        occurrence_count,
        success_rate,
        CAST(occurrence_count AS REAL) / (SELECT SUM(occurrence_count) FROM patterns WHERE pattern_sequence = p.pattern_sequence) AS probability
    FROM patterns p
    """
}