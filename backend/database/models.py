#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veritabanı Modelleri
------------------
SQLite tablo modelleri ve ORM sınıfları.
Bu modül ORM kullanmak istenildiğinde aktif edilebilir.
"""

# Bu projede şu anda doğrudan SQL kullanılmaktadır.
# İleride SQLAlchemy gibi bir ORM kullanılacaksa, modeller burada tanımlanabilir.

"""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class GameResult(Base):
    __tablename__ = 'game_results'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    result = Column(String(1), nullable=False)
    previous_pattern = Column(String)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    
    session = relationship("Session", back_populates="results")
    predictions = relationship("Prediction", back_populates="game_result")
    features = relationship("DeepLearningFeature", back_populates="game_result")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    game_result_id = Column(Integer, ForeignKey('game_results.id'))
    algorithm_id = Column(Integer, ForeignKey('algorithms.id'))
    predicted_result = Column(String(1), nullable=False)
    is_correct = Column(Boolean)
    confidence_score = Column(Float)
    
    game_result = relationship("GameResult", back_populates="predictions")
    algorithm = relationship("Algorithm", back_populates="predictions")

class Algorithm(Base):
    __tablename__ = 'algorithms'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String)
    current_accuracy = Column(Float)
    weight = Column(Float)
    last_updated = Column(DateTime, default=datetime.now)
    
    predictions = relationship("Prediction", back_populates="algorithm")
    performances = relationship("AlgorithmPerformance", back_populates="algorithm")

class Pattern(Base):
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True)
    pattern_sequence = Column(String, nullable=False)
    occurrence_count = Column(Integer)
    next_result = Column(String(1))
    success_rate = Column(Float)

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    file_path = Column(String)
    accuracy = Column(Float)
    is_active = Column(Boolean)
    
    features = relationship("DeepLearningFeature", back_populates="model_version")

class Session(Base):
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime)
    total_games = Column(Integer)
    win_rate = Column(Float)
    
    results = relationship("GameResult", back_populates="session")

class AlgorithmPerformance(Base):
    __tablename__ = 'algorithm_performance'
    
    id = Column(Integer, primary_key=True)
    algorithm_id = Column(Integer, ForeignKey('algorithms.id'))
    evaluation_date = Column(DateTime, default=datetime.now)
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)
    accuracy = Column(Float)
    
    algorithm = relationship("Algorithm", back_populates="performances")

class DeepLearningFeature(Base):
    __tablename__ = 'deep_learning_features'
    
    id = Column(Integer, primary_key=True)
    game_result_id = Column(Integer, ForeignKey('game_results.id'))
    feature_vector = Column(String)  # JSON string
    created_at = Column(DateTime, default=datetime.now)
    model_version_id = Column(Integer, ForeignKey('model_versions.id'))
    
    game_result = relationship("GameResult", back_populates="features")
    model_version = relationship("ModelVersion", back_populates="features")
"""