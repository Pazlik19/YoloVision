import os
import cv2
import yaml
import json
from tqdm import tqdm
from pathlib import Path
# Подключаем твои классы (предположим, они в этом же файле или импортированы)
from TextDetection2 import SymbolClassifier 
from Segmentation import Segmentation

# --- НАСТРОЙКИ ---
TARGET_DIR = Path(__file__).parent.parent / "Models" 
INPUT_DIR = 'Data'
OUTPUT_DIR = 'yolo_dataset'
CONF_THRESHOLD = 20.0 # Порог для ResNet

def generate():
    # 1. Инициализация
    # Укажи путь к своей обученной ResNet и json с именами
    classifier = SymbolClassifier(model_path=TARGET_DIR/'best_resnet_model.pth', config_path=TARGET_DIR/'class_names.json')
    # Используем YOLO для сегментации (можно взять стандартную или свою)
    segmentator = Segmentation(model_path=TARGET_DIR/'best.pt') 

    # Загружаем имена классов для YOLO из конфига классификатора
    class_names = classifier.class_names # Должен быть список или dict
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

    # Создаем data.yaml
    data_yaml = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images',
        'val': 'images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(img_path)
        if image is None: continue
        
        h, w = image.shape[:2]
        
        # Шаг 1: Сегментация (YOLO находит объекты)
        crops, bboxes = segmentator.get_objects(image)
        
        tqdm(bboxes)
        yolo_labels = []
        
        # Шаг 2: Классификация (ResNet проверяет каждый кроп)
        for crop, box in zip(crops, bboxes):
            label_name, confidence = classifier.predict(crop)
            
            if confidence >= CONF_THRESHOLD:
                x1, y1, x2, y2 = box
                # Конвертация в формат YOLO (0-1)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                class_id = class_to_id[label_name]
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Шаг 3: Сохранение
        if yolo_labels:
            cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}", image)
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(f"{OUTPUT_DIR}/labels/{txt_name}", 'w') as f:
                f.write("\n".join(yolo_labels))

if __name__ == "__main__":
    generate()