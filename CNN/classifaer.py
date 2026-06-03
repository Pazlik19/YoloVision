import os
import shutil
from tqdm import tqdm
from TextDetection2 import SymbolClassifier
from pathlib import Path

# Убедись, что класс SymbolClassifier определен выше или импортирован

def sort_images_with_threshold(source_dir, output_root, model_path, config_path, threshold=80.0):
    # 1. Инициализируем модель
    classifier = SymbolClassifier(model_path=model_path, config_path=config_path)
    
    # 2. Создаем корневую папку и папку для "слабых" предсказаний
    low_conf_dir = os.path.join(output_root, "low_confidence2")
    for path in [output_root, low_conf_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"Найдено изображений: {len(files)}")
    print(f"Порог уверенности: {threshold}%")

    for filename in tqdm(files):
        file_path = os.path.join(source_dir, filename)
        
        try:
            # 3. Получаем предсказание и уверенность
            label, confidence = classifier.predict(file_path)
            
            # 4. Проверка порога
            if confidence >= threshold:
                # Если всё ок, кладем в папку с названием класса
                target_folder_name = f"latter_{label}"
                target_folder_path = os.path.join(output_root, target_folder_name)
            else:
                # Если нейросеть сомневается, отправляем в спецпапку
                target_folder_path = low_conf_dir
            
            # Создаем папку, если её еще нет
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)
            
            # 5. Копируем файл
            shutil.copy(file_path, os.path.join(target_folder_path, filename))
            
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")

# --- Настройки ---
CURENT_PATH = Path(__file__).parent
SOURCE_FOLDER = 'detected_objects'
RESULT_FOLDER = 'sorted_symbols'
MODEL_WEIGHTS = CURENT_PATH.parent/'Models' / 'best_resnet_model.pth'
CONFIG_JSON = CURENT_PATH.parent/'Models' / 'class_names.json'
CONFIDENCE_THRESHOLD = 70.0  # Установи свой порог здесь

if __name__ == "__main__":
    sort_images_with_threshold(
        SOURCE_FOLDER, 
        RESULT_FOLDER, 
        MODEL_WEIGHTS, 
        CONFIG_JSON, 
        threshold=CONFIDENCE_THRESHOLD
    )
    print(f"\nСортировка завершена! Проверь папку {RESULT_FOLDER}/low_confidence на наличие ошибок.")