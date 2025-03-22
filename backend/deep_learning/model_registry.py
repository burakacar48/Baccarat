#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Kayıt Modülü
-----------------
LSTM modellerinin kaydedilmesini, yüklenmesini ve yönetilmesini sağlar.
"""

import os
import logging
import json
import shutil
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Model Kayıt Sınıfı
    
    LSTM modellerinin kaydedilmesini, yüklenmesini ve yönetilmesini sağlar.
    """
    
    def __init__(self, db_manager, base_path='models'):
        """
        ModelRegistry sınıfını başlatır
        
        Args:
            db_manager: Veritabanı yönetici nesnesi
            base_path (str): Modellerin kaydedileceği temel dizin
        """
        self.db_manager = db_manager
        self.base_path = base_path
        
        # Dizin kontrolü
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        logger.info(f"Model kayıt modülü başlatıldı: {base_path}")
    
    def save_model(self, model, model_type="LSTM", version=None, metrics=None, set_active=True):
        """
        Modeli kaydeder
        
        Args:
            model: LSTM model nesnesi
            model_type (str): Model tipi
            version (str, optional): Model versiyonu, None ise otomatik oluşturulur
            metrics (dict, optional): Model performans metrikleri
            set_active (bool): Model aktif olarak ayarlansın mı
        
        Returns:
            str: Kaydedilen model dosya yolu
        """
        try:
            # Versiyon kontrolü
            if version is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                version = f"v{timestamp}"
            
            # Model dizini
            model_dir = os.path.join(self.base_path, model_type.lower())
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Model dosya yolu
            model_filename = f"{model_type.lower()}_{version}.h5"
            model_path = os.path.join(model_dir, model_filename)
            
            # Modeli kaydet
            model.save_model(model_path)
            
            # Meta veri dosya yolu
            meta_filename = f"{model_type.lower()}_{version}_meta.json"
            meta_path = os.path.join(model_dir, meta_filename)
            
            # Meta veriyi hazırla
            meta_data = {
                'model_type': model_type,
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'output_size': model.output_size,
                'metrics': metrics or {},
                'file_path': model_path
            }
            
            # Meta veriyi kaydet
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            # Model kontrol dosyası (hash)
            self._create_hash_file(model_path)
            
            # Veritabanına kaydet
            accuracy = metrics.get('accuracy', 0.0) if metrics else 0.0
            self.db_manager.save_model_version(
                model_type=model_type,
                file_path=model_path,
                accuracy=accuracy,
                is_active=set_active
            )
            
            # Aktif model sembolik bağlantısı
            if set_active:
                self._set_active_model(model_type, model_path)
            
            logger.info(f"Model kaydedildi: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {str(e)}")
            return None
    
    def load_model(self, model, version=None, model_type="LSTM"):
        """
        Modeli yükler
        
        Args:
            model: LSTM model nesnesi
            version (str, optional): Yüklenecek model versiyonu, None ise aktif model
            model_type (str): Model tipi
        
        Returns:
            bool: Yükleme başarılı ise True, değilse False
        """
        try:
            model_path = None
            
            # Versiyon belirtilmişse o versiyonu yükle
            if version is not None:
                model_filename = f"{model_type.lower()}_{version}.h5"
                model_dir = os.path.join(self.base_path, model_type.lower())
                model_path = os.path.join(model_dir, model_filename)
                
                if not os.path.exists(model_path):
                    logger.error(f"Model bulunamadı: {model_path}")
                    return False
            else:
                # Aktif modeli getir
                active_model = self.db_manager.get_active_model(model_type)
                
                if active_model:
                    model_path = active_model['file_path']
                    
                    # Dosya varlığını kontrol et
                    if not os.path.exists(model_path):
                        logger.error(f"Aktif model dosyası bulunamadı: {model_path}")
                        return False
                else:
                    logger.error(f"Aktif {model_type} modeli bulunamadı")
                    return False
            
            # Model bütünlüğünü doğrula
            if not self._verify_hash(model_path):
                logger.error(f"Model bütünlük doğrulaması başarısız: {model_path}")
                return False
            
            # Modeli yükle
            result = model.load_model(model_path)
            
            if result:
                logger.info(f"Model başarıyla yüklendi: {model_path}")
            else:
                logger.error(f"Model yükleme başarısız: {model_path}")
            
            return result
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
    
    def get_model_info(self, model_type="LSTM", version=None):
        """
        Model bilgilerini getirir
        
        Args:
            model_type (str): Model tipi
            version (str, optional): Model versiyonu, None ise aktif model
        
        Returns:
            dict: Model bilgileri, bulunamazsa None
        """
        try:
            # Versiyon belirtilmişse o versiyonu getir
            if version is not None:
                meta_filename = f"{model_type.lower()}_{version}_meta.json"
                model_dir = os.path.join(self.base_path, model_type.lower())
                meta_path = os.path.join(model_dir, meta_filename)
                
                if not os.path.exists(meta_path):
                    logger.error(f"Model meta verisi bulunamadı: {meta_path}")
                    return None
                
                # Meta veriyi oku
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                
                return meta_data
            else:
                # Aktif modeli getir
                active_model = self.db_manager.get_active_model(model_type)
                
                if active_model:
                    # Dosya yolundan meta veriyi bul
                    model_path = active_model['file_path']
                    meta_path = model_path.replace('.h5', '_meta.json')
                    
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            meta_data = json.load(f)
                        
                        return meta_data
                    else:
                        # Meta veri yoksa veritabanı bilgilerini döndür
                        return active_model
                else:
                    logger.error(f"Aktif {model_type} modeli bulunamadı")
                    return None
        except Exception as e:
            logger.error(f"Model bilgisi getirme hatası: {str(e)}")
            return None
    
    def list_models(self, model_type="LSTM"):
        """
        Mevcut model versiyonlarını listeler
        
        Args:
            model_type (str): Model tipi
        
        Returns:
            list: Model versiyonları listesi
        """
        try:
            model_dir = os.path.join(self.base_path, model_type.lower())
            
            if not os.path.exists(model_dir):
                logger.warning(f"Model dizini bulunamadı: {model_dir}")
                return []
            
            # Meta veri dosyalarını bul
            meta_files = [f for f in os.listdir(model_dir) if f.endswith('_meta.json')]
            
            # Meta verileri oku
            models = []
            for meta_file in meta_files:
                meta_path = os.path.join(model_dir, meta_file)
                
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                
                # Aktif model bilgisini ekle
                active_model = self.db_manager.get_active_model(model_type)
                is_active = False
                
                if active_model and 'file_path' in meta_data:
                    is_active = active_model['file_path'] == meta_data['file_path']
                
                meta_data['is_active'] = is_active
                models.append(meta_data)
            
            # Zaman damgasına göre sırala (en yeni başta)
            models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return models
        except Exception as e:
            logger.error(f"Model listesi getirme hatası: {str(e)}")
            return []
    
    def delete_model(self, version, model_type="LSTM"):
        """
        Belirli bir model versiyonunu siler
        
        Args:
            version (str): Model versiyonu
            model_type (str): Model tipi
        
        Returns:
            bool: Silme başarılı ise True, değilse False
        """
        try:
            model_dir = os.path.join(self.base_path, model_type.lower())
            model_filename = f"{model_type.lower()}_{version}.h5"
            meta_filename = f"{model_type.lower()}_{version}_meta.json"
            hash_filename = f"{model_type.lower()}_{version}.hash"
            
            model_path = os.path.join(model_dir, model_filename)
            meta_path = os.path.join(model_dir, meta_filename)
            hash_path = os.path.join(model_dir, hash_filename)
            
            # Aktif model mi kontrol et
            active_model = self.db_manager.get_active_model(model_type)
            
            if active_model and active_model['file_path'] == model_path:
                logger.error(f"Aktif model silinemez: {model_path}")
                return False
            
            # Dosyaları sil
            files_to_delete = [model_path, meta_path, hash_path]
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Dosya silindi: {file_path}")
            
            # Veritabanından kaldır
            # NOT: Burada model_versions tablosundan silme işlemi yapılabilir
            
            logger.info(f"Model silindi: {model_type} {version}")
            return True
        except Exception as e:
            logger.error(f"Model silme hatası: {str(e)}")
            return False
    
    def compare_models(self, version1, version2, model_type="LSTM"):
        """
        İki model versiyonunu karşılaştırır
        
        Args:
            version1 (str): Birinci model versiyonu
            version2 (str): İkinci model versiyonu
            model_type (str): Model tipi
        
        Returns:
            dict: Karşılaştırma sonuçları
        """
        try:
            # Model bilgilerini getir
            info1 = self.get_model_info(model_type, version1)
            info2 = self.get_model_info(model_type, version2)
            
            if not info1 or not info2:
                logger.error("Model bilgileri getirilemedi")
                return None
            
            # Metrikleri karşılaştır
            metrics1 = info1.get('metrics', {})
            metrics2 = info2.get('metrics', {})
            
            # Karşılaştırma sonuçları
            comparison = {
                'model1': {
                    'version': version1,
                    'timestamp': info1.get('timestamp', ''),
                    'metrics': metrics1
                },
                'model2': {
                    'version': version2,
                    'timestamp': info2.get('timestamp', ''),
                    'metrics': metrics2
                },
                'differences': {}
            }
            
            # Tüm metrikleri karşılaştır
            all_metrics = set(metrics1.keys()).union(set(metrics2.keys()))
            
            for metric in all_metrics:
                value1 = metrics1.get(metric, None)
                value2 = metrics2.get(metric, None)
                
                if value1 is not None and value2 is not None:
                    difference = value2 - value1
                    percent_change = (difference / value1) * 100 if value1 != 0 else float('inf')
                    
                    comparison['differences'][metric] = {
                        'difference': difference,
                        'percent_change': percent_change
                    }
            
            return comparison
        except Exception as e:
            logger.error(f"Model karşılaştırma hatası: {str(e)}")
            return None
    
    def _set_active_model(self, model_type, model_path):
        """
        Belirli bir modeli aktif olarak ayarlar
        
        Args:
            model_type (str): Model tipi
            model_path (str): Model dosya yolu
        """
        try:
            # Sembolik bağlantı için hedef
            model_dir = os.path.join(self.base_path, model_type.lower())
            active_link = os.path.join(model_dir, f"{model_type.lower()}_active.h5")
            
            # Önceki sembolik bağlantıyı kaldır
            if os.path.exists(active_link):
                if os.path.islink(active_link):
                    os.unlink(active_link)
                else:
                    os.remove(active_link)
            
            # Yeni sembolik bağlantı oluştur
            # Windows'ta sembolik bağlantı desteklenmediği için kopyalama da yapabiliriz
            try:
                os.symlink(model_path, active_link)
            except (OSError, AttributeError):
                # Sembolik bağlantı oluşturulamazsa dosyayı kopyala
                shutil.copy2(model_path, active_link)
            
            logger.info(f"Aktif model güncellendi: {model_path}")
        except Exception as e:
            logger.error(f"Aktif model ayarlama hatası: {str(e)}")
    
    def _create_hash_file(self, file_path):
        """
        Dosya için hash değerini hesaplar ve kaydeder
        
        Args:
            file_path (str): Dosya yolu
        """
        try:
            # Dosya hash'ini hesapla
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            hash_value = hash_md5.hexdigest()
            
            # Hash dosyasını kaydet
            hash_path = file_path.replace('.h5', '.hash')
            with open(hash_path, 'w') as f:
                f.write(hash_value)
            
            logger.debug(f"Hash dosyası oluşturuldu: {hash_path}")
        except Exception as e:
            logger.error(f"Hash dosyası oluşturma hatası: {str(e)}")
    
    def _verify_hash(self, file_path):
        """
        Dosya hash'ini doğrular
        
        Args:
            file_path (str): Dosya yolu
        
        Returns:
            bool: Hash doğruysa True, değilse False
        """
        try:
            # Hash dosyasını kontrol et
            hash_path = file_path.replace('.h5', '.hash')
            
            if not os.path.exists(hash_path):
                logger.warning(f"Hash dosyası bulunamadı: {hash_path}")
                return True  # Hash dosyası yoksa doğrulama atlansın
            
            # Kayıtlı hash'i oku
            with open(hash_path, 'r') as f:
                saved_hash = f.read().strip()
            
            # Mevcut hash'i hesapla
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            current_hash = hash_md5.hexdigest()
            
            # Hash'leri karşılaştır
            is_valid = saved_hash == current_hash
            
            if not is_valid:
                logger.warning(f"Hash uyuşmazlığı: {file_path}")
            
            return is_valid
        except Exception as e:
            logger.error(f"Hash doğrulama hatası: {str(e)}")
            return False