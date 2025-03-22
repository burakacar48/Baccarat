#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loglama Konfigürasyonu
--------------------
Uygulama loglama ayarlarını yapılandırır.
"""

import os
import logging
import logging.handlers
from datetime import datetime

def configure_logging(log_file=None, log_level=logging.INFO, debug_mode=False):
    """
    Uygulama loglama ayarlarını yapılandırır
    
    Args:
        log_file (str, optional): Log dosyası yolu
        log_level (int): Log seviyesi (logging.DEBUG, logging.INFO, vb.)
        debug_mode (bool): Debug modu etkin mi
    """
    # Log klasörünü kontrol et/oluştur
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # Root logger'ı ayarla
    root_logger = logging.getLogger()
    
    # Mevcut handler'ları temizle
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Debug modu etkinse log seviyesini DEBUG'a ayarla
    if debug_mode:
        log_level = logging.DEBUG
    
    root_logger.setLevel(log_level)
    
    # Format tanımla
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Konsol çıktısı için handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # Dosya çıktısı için handler (opsiyonel)
    if log_file:
        # RotatingFileHandler: Dosya boyutu sınırlı ve rotasyon yapabilen handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,  # En fazla 5 yedek dosya
            encoding='utf-8'
        )
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
    
    # Debug modu için ek ayarlar
    if debug_mode:
        # Üçüncü parti kütüphanelerin gereksiz loglarını kapat
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        # Bilgilendirme mesajı
        root_logger.debug("Debug modu etkin")
    
    return root_logger