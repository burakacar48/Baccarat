import os

def fix_db_manager():
    # db_manager.py dosyasının tam yolunu belirle
    db_manager_path = os.path.abspath("db_manager.py")
    
    # Dosyanın var olup olmadığını kontrol et
    if not os.path.isfile(db_manager_path):
        print(f"HATA: {db_manager_path} dosyası bulunamadı!")
        return
    
    # Dosyayı oku
    with open(db_manager_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # SESSIONS tablosu oluşturma kodunu bul ve sil
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if "# SESSIONS tablosu" in line:
            start_index = i
        elif start_index is not None and ")))" in line:
            end_index = i
            break
    
    if start_index is not None and end_index is not None:
        del lines[start_index:end_index+1]
    
    # _create_tables metodunun içine SESSIONS tablosu oluşturma kodunu ekle
    create_tables_method_index = lines.index("    def _create_tables(self, cursor):\n")
    lines.insert(create_tables_method_index + 4, '''
    # SESSIONS tablosu
    cursor.execute(\'\'\'
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT DEFAULT CURRENT_TIMESTAMP
    )
    \'\'\')
    ''')
    
    # Değişiklikleri dosyaya kaydet
    with open(db_manager_path, "w", encoding="utf-8") as file:  
        file.writelines(lines)
    
    print(f"{db_manager_path} dosyası başarıyla güncellendi.")

# Betiği çalıştır
if __name__ == "__main__":
    fix_db_manager()