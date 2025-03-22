#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Şema Modülü
-------------
REST API'lerin girdi ve çıktı şemalarını tanımlar.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class ResultInput(BaseModel):
    """Oyun sonucu girişi"""
    result: str = Field(..., description="Oyun sonucu (P/B/T)")
    timestamp: Optional[str] = Field(None, description="Zaman damgası (ISO 8601)")
    session_id: Optional[int] = Field(None, description="Oturum ID")
    
    @validator('result')
    def validate_result(cls, v):
        if v not in ['P', 'B', 'T']:
            raise ValueError("Sonuç P, B veya T olmalıdır")
        return v

class BulkResultInput(BaseModel):
    """Toplu sonuç girişi"""
    results: str = Field(..., description="Sonuç dizisi (örn: 'PPBBTPBPBP')")
    
    @validator('results')
    def validate_results(cls, v):
        if not all(c in ['P', 'B', 'T'] for c in v):
            raise ValueError("Sonuçlar yalnızca P, B, T karakterlerini içermelidir")
        return v

class PredictionResult(BaseModel):
    """Tahmin sonucu"""
    prediction: str = Field(..., description="Tahmin edilen sonuç (P/B/T)")
    confidence: float = Field(..., description="Güven skoru (0-1)")
    timestamp: str = Field(..., description="Zaman damgası")
    algorithms: List[Dict[str, Any]] = Field([], description="Algoritma tahminleri")
    contributions: Dict[str, float] = Field({}, description="Algoritma katkıları")
    execution_time: float = Field(0.0, description="Yürütme süresi (saniye)")
    details: Dict[str, float] = Field({}, description="Sonuç detayları")

class ResultResponse(BaseModel):
    """Oyun sonucu ve tahmin karşılaştırması"""
    id: int = Field(..., description="Sonuç ID")
    result: str = Field(..., description="Gerçek sonuç (P/B/T)")
    prediction: str = Field(..., description="Tahmin edilen sonuç (P/B/T)")
    is_correct: bool = Field(..., description="Tahmin doğru mu")
    confidence: float = Field(..., description="Güven skoru")
    timestamp: str = Field(..., description="Zaman damgası")

class AlgorithmPerformance(BaseModel):
    """Algoritma performans bilgileri"""
    id: int = Field(..., description="Algoritma ID")
    name: str = Field(..., description="Algoritma adı")
    total_predictions: int = Field(..., description="Toplam tahmin sayısı")
    correct_predictions: int = Field(..., description="Doğru tahmin sayısı")
    accuracy: float = Field(..., description="Doğruluk oranı")
    weight: float = Field(..., description="Ağırlık")
    last_updated: str = Field(..., description="Son güncelleme")

class OptimizationResult(BaseModel):
    """Ağırlık optimizasyonu sonucu"""
    old_weight: float = Field(..., description="Eski ağırlık")
    new_weight: float = Field(..., description="Yeni ağırlık")
    accuracy: Optional[float] = Field(None, description="Doğruluk oranı")
    message: Optional[str] = Field(None, description="Mesaj")

class OptimizationResponse(BaseModel):
    """Ağırlık optimizasyonu yanıtı"""
    results: Dict[str, OptimizationResult] = Field(..., description="Optimizasyon sonuçları")
    timestamp: str = Field(..., description="Zaman damgası")
    strategy: str = Field(..., description="Optimizasyon stratejisi")

class ModelInfo(BaseModel):
    """Model bilgileri"""
    id: int = Field(..., description="Model ID")
    model_type: str = Field(..., description="Model tipi")
    created_at: str = Field(..., description="Oluşturulma zamanı")
    file_path: str = Field(..., description="Dosya yolu")
    accuracy: float = Field(..., description="Doğruluk oranı")
    is_active: bool = Field(..., description="Aktif mi")
    metrics: Optional[Dict[str, Any]] = Field({}, description="Model metrikleri")

class SessionInfo(BaseModel):
    """Oturum bilgileri"""
    id: int = Field(..., description="Oturum ID")
    start_time: str = Field(..., description="Başlangıç zamanı")
    end_time: Optional[str] = Field(None, description="Bitiş zamanı")
    total_games: int = Field(0, description="Toplam oyun sayısı")
    win_rate: float = Field(0.0, description="Kazanç oranı")

class LoginRequest(BaseModel):
    """Giriş isteği"""
    username: str = Field(..., description="Kullanıcı adı")
    password: str = Field(..., description="Şifre")

class LoginResponse(BaseModel):
    """Giriş yanıtı"""
    access_token: str = Field(..., description="Erişim jetonu")
    token_type: str = Field("bearer", description="Jeton tipi")
    expires_in: int = Field(3600, description="Süre sonu (saniye)")

class Error(BaseModel):
    """Hata yanıtı"""
    error: str = Field(..., description="Hata kodu")
    message: str = Field(..., description="Hata mesajı")
    timestamp: str = Field(datetime.now().isoformat(), description="Zaman damgası")

class Health(BaseModel):
    """Sağlık durumu"""
    status: str = Field("ok", description="Durum")
    version: str = Field(..., description="Versiyon")
    timestamp: str = Field(datetime.now().isoformat(), description="Zaman damgası")
    uptime: float = Field(..., description="Çalışma süresi (saniye)")
    components: Dict[str, str] = Field({}, description="Bileşen durumları")