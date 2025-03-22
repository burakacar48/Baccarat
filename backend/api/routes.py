#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Rota Tanımları
----------------
REST API rotalarını ve endpoint'leri tanımlar.
"""

import os
import time
import logging
import secrets
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException, status, Body, Query, Path, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt

# Yerel modülleri import et
from .schema import (
    ResultInput, BulkResultInput, PredictionResult, ResultResponse,
    AlgorithmPerformance, OptimizationResponse, ModelInfo, SessionInfo,
    LoginRequest, LoginResponse, Error, Health
)

logger = logging.getLogger(__name__)

# Uygulamanın başlangıç zamanı
start_time = time.time()

# JWT ayarları
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# OAuth2 şema
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

def create_app(prediction_engine, performance_tracker, weight_optimizer, db_manager, model_registry=None):
    """
    FastAPI uygulamasını oluşturur ve rotaları tanımlar
    
    Args:
        prediction_engine: Tahmin motoru nesnesi
        performance_tracker: Performans izleyici nesnesi
        weight_optimizer: Ağırlık optimizasyonu nesnesi
        db_manager: Veritabanı yönetici nesnesi
        model_registry: Model kayıt nesnesi
    
    Returns:
        FastAPI: FastAPI uygulaması
    """
    # FastAPI uygulaması
    app = FastAPI(
        title="Baccarat Tahmin Sistemi API",
        description="14 farklı algoritma ve derin öğrenme entegrasyonu içeren baccarat tahmin sistemi API'si",
        version="1.0.0",
    )
    
    # CORS ayarları
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Kullanıcı kimlik doğrulama
    fake_users_db = {
        "admin": {
            "username": "admin",
            "password": "admin123",  # Gerçek uygulamada hash kullanın
            "disabled": False,
        }
    }
    
    def authenticate_user(username: str, password: str):
        user = fake_users_db.get(username)
        if not user:
            return False
        if user["password"] != password:  # Gerçek uygulamada hash kullanın
            return False
        return user
    
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    async def get_current_user(token: str = Depends(oauth2_scheme)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except jwt.PyJWTError:
            raise credentials_exception
        
        user = fake_users_db.get(username)
        if user is None:
            raise credentials_exception
        
        return User(**user)
    
    async def get_current_active_user(current_user: User = Depends(get_current_user)):
        if current_user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    # Oturum açma endpoint'i
    @app.post("/token", response_model=LoginResponse)
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    # Sağlık kontrolü endpoint'i
    @app.get("/health", response_model=Health)
    async def health_check():
        uptime = time.time() - start_time
        
        # Bileşen durumlarını kontrol et
        components = {
            "database": "ok" if db_manager and db_manager.connection else "error",
            "prediction_engine": "ok" if prediction_engine else "error",
            "performance_tracker": "ok" if performance_tracker else "error"
        }
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime,
            "components": components
        }
    
    # Tahmin endpoint'i
    @app.get("/predict", response_model=PredictionResult)
    async def get_prediction(
        current_user: User = Depends(get_current_active_user),
        limit: int = Query(20, description="Son kaç sonucun kullanılacağı")
    ):
        # Son N sonucu getir
        last_results = db_manager.get_last_n_results(limit)
        
        if not last_results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Yeterli sonuç verisi yok"
            )
        
        # Sonuçları düzenle
        results = [result['result'] for result in last_results]
        results.reverse()  # En eskiden en yeniye sırala
        
        # Tahmin yap
        prediction = prediction_engine.predict({'last_results': results})
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tahmin yapılamadı"
            )
        
        return prediction
    
    # Sonuç girişi endpoint'i
    @app.post("/results", response_model=ResultResponse)
    async def add_result(
        result_input: ResultInput,
        current_user: User = Depends(get_current_active_user)
    ):
        # Önce tahmin yap
        last_results = db_manager.get_last_n_results(20)
        
        if not last_results:
            # İlk sonuç girişiyse tahmin yapma
            result_id = db_manager.save_result(
                result=result_input.result,
                timestamp=result_input.timestamp or datetime.now().isoformat(),
                session_id=result_input.session_id
            )
            
            return {
                "id": result_id,
                "result": result_input.result,
                "prediction": None,
                "is_correct": False,
                "confidence": 0.0,
                "timestamp": result_input.timestamp or datetime.now().isoformat()
            }
        
        # Sonuçları düzenle
        results = [result['result'] for result in last_results]
        results.reverse()  # En eskiden en yeniye sırala
        
        # Tahmin yap
        prediction = prediction_engine.predict({'last_results': results})
        
        # Sonucu kaydet
        result_id = db_manager.save_result(
            result=result_input.result,
            timestamp=result_input.timestamp or datetime.now().isoformat(),
            session_id=result_input.session_id
        )
        
        # Sonuç ve tahmin bilgilerini dön
        is_correct = prediction['prediction'] == result_input.result if prediction else False
        confidence = prediction['confidence'] if prediction else 0.0
        
        return {
            "id": result_id,
            "result": result_input.result,
            "prediction": prediction['prediction'] if prediction else None,
            "is_correct": is_correct,
            "confidence": confidence,
            "timestamp": result_input.timestamp or datetime.now().isoformat()
        }
    
    # Toplu sonuç girişi endpoint'i
    @app.post("/results/bulk", response_model=List[ResultResponse])
    async def add_bulk_results(
        bulk_input: BulkResultInput,
        current_user: User = Depends(get_current_active_user)
    ):
        responses = []
        
        for result_char in bulk_input.results:
            # Her bir sonuç için aynı işlemi yap
            result_input = ResultInput(result=result_char)
            response = await add_result(result_input, current_user)
            responses.append(response)
        
        return responses
    
    # Son sonuçları getirme endpoint'i
    @app.get("/results", response_model=List[ResultResponse])
    async def get_results(
        current_user: User = Depends(get_current_active_user),
        limit: int = Query(20, description="Getirilecek sonuç sayısı"),
        skip: int = Query(0, description="Atlanacak sonuç sayısı")
    ):
        # Son N sonucu getir
        results = db_manager.get_results(limit=limit, skip=skip)
        
        if not results:
            return []
        
        # Sonuçları düzenle
        formatted_results = []
        
        for result in results:
            # Tahmin bilgisini getir
            prediction = db_manager.get_prediction_for_result(result['id'])
            
            formatted_results.append({
                "id": result['id'],
                "result": result['result'],
                "prediction": prediction['predicted_result'] if prediction else None,
                "is_correct": prediction['is_correct'] if prediction else False,
                "confidence": prediction['confidence_score'] if prediction else 0.0,
                "timestamp": result['timestamp']
            })
        
        return formatted_results
    
    # Algoritma performanslarını getirme endpoint'i
    @app.get("/algorithms/performance", response_model=List[AlgorithmPerformance])
    async def get_algorithm_performance(
        current_user: User = Depends(get_current_active_user),
        days: int = Query(30, description="Kaç günlük performans getirileceği")
    ):
        # Tüm algoritmaları getir
        algorithms = db_manager.get_all_algorithms()
        
        if not algorithms:
            return []
        
        # Her algoritma için performans bilgilerini getir
        performance_data = []
        
        for algorithm in algorithms:
            algorithm_id = algorithm['id']
            
            # Son N günlük performansı getir
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            performance = db_manager.get_algorithm_performance_since_date(algorithm_id, start_date)
            
            if performance:
                performance_data.append({
                    "id": algorithm_id,
                    "name": algorithm['name'],
                    "total_predictions": performance['total_predictions'],
                    "correct_predictions": performance['correct_predictions'],
                    "accuracy": performance['accuracy'],
                    "weight": algorithm['weight'],
                    "last_updated": algorithm['last_updated']
                })
        
        return performance_data
    
    # Ağırlık optimizasyonu endpoint'i
    @app.post("/algorithms/optimize", response_model=OptimizationResponse)
    async def optimize_weights(
        current_user: User = Depends(get_current_active_user),
        strategy: str = Query("performance", description="Optimizasyon stratejisi"),
        days: int = Query(7, description="Kaç günlük verinin kullanılacağı")
    ):
        # Ağırlıkları optimize et
        results = weight_optimizer.optimize_weights(strategy=strategy, days=days)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ağırlık optimizasyonu başarısız oldu"
            )
        
        return {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy
        }
    
    # Model eğitimi endpoint'i
    @app.post("/models/train", response_model=ModelInfo)
    async def train_model(
        current_user: User = Depends(get_current_active_user),
        model_type: str = Query("LSTM", description="Model tipi"),
        epochs: int = Query(50, description="Eğitim döngüsü sayısı"),
        batch_size: int = Query(32, description="Batch boyutu"),
        force: bool = Query(False, description="Zorla yeniden eğit")
    ):
        if model_registry is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Model kayıt modülü etkin değil"
            )
        
        # LSTM modelini al
        lstm_model = prediction_engine.deep_learning_model
        
        if not lstm_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LSTM modeli bulunamadı"
            )
        
        from backend.deep_learning.training import ModelTrainer
        model_trainer = ModelTrainer(lstm_model, db_manager)
        
        # Modeli eğit
        training_history = model_trainer.train_model(epochs=epochs, batch_size=batch_size, force=force)
        
        if not training_history:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model eğitimi başarısız oldu"
            )
        
        # Eğitim metriklerini hesapla
        if isinstance(training_history, dict) and 'accuracy' in training_history:
            accuracy = training_history['accuracy'][-1]
        else:
            accuracy = 0.0
        
        # Modeli kaydet
        metrics = {
            "accuracy": accuracy,
            "epochs": epochs,
            "batch_size": batch_size
        }
        
        # Veritabanında model versiyonunu güncelle
        active_model = db_manager.get_active_model(model_type)
        
        if active_model:
            return {
                "id": active_model['id'],
                "model_type": model_type,
                "created_at": datetime.now().isoformat(),
                "file_path": active_model['file_path'],
                "accuracy": accuracy,
                "is_active": True,
                "metrics": metrics
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model veritabanında bulunamadı"
            )
    
    # Model bilgilerini getirme endpoint'i
    @app.get("/models", response_model=List[ModelInfo])
    async def get_models(
        current_user: User = Depends(get_current_active_user),
        model_type: str = Query("LSTM", description="Model tipi")
    ):
        if model_registry is None:
            # Model kayıt modülü yoksa veritabanından getir
            models = db_manager.get_model_versions(model_type)
            
            if not models:
                return []
            
            return models
        else:
            # Model kayıt modülü varsa oradan getir
            models = model_registry.list_models(model_type)
            
            if not models:
                return []
            
            # Modelleri formatlı hale getir
            formatted_models = []
            
            for model in models:
                formatted_models.append({
                    "id": model.get('id', 0),
                    "model_type": model.get('model_type', model_type),
                    "created_at": model.get('timestamp', ''),
                    "file_path": model.get('file_path', ''),
                    "accuracy": model.get('metrics', {}).get('accuracy', 0.0),
                    "is_active": model.get('is_active', False),
                    "metrics": model.get('metrics', {})
                })
            
            return formatted_models
    
    # Oturum bilgilerini getirme endpoint'i
    @app.get("/sessions", response_model=List[SessionInfo])
    async def get_sessions(
        current_user: User = Depends(get_current_active_user),
        limit: int = Query(10, description="Getirilecek oturum sayısı")
    ):
        # Oturumları getir
        sessions = db_manager.get_sessions(limit)
        
        if not sessions:
            return []
        
        return sessions
    
    # Yeni oturum oluşturma endpoint'i
    @app.post("/sessions", response_model=SessionInfo)
    async def create_session(
        current_user: User = Depends(get_current_active_user)
    ):
        # Yeni oturum oluştur
        session_id = db_manager.create_session()
        
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Oturum oluşturulamadı"
            )
        
        # Oturum bilgilerini getir
        session = db_manager.get_session(session_id)
        
        return session
    
    # Hata yakalama
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return Error(
            error=str(exc.status_code),
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.exception("Beklenmeyen hata")
        return Error(
            error="500",
            message="Sunucu hatası",
            timestamp=datetime.now().isoformat()
        )
    
    return app