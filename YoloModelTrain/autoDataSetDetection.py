import cv2
import os
import torch
import json
import numpy as np
from tqdm import tqdm
from TextDetection2 import SymbolClassifier
from Segmentation import Segmentation as seg
import yaml

# --- НАСТРОЙКИ ---
INPUT_DIR = 'Data'               # Папка с исходными фото
OUTPUT_DIR = 'yolo_dataset'      # Папка для датасета
CONF_THRESHOLD = 85.0            # Повысил порог для более чистой разметки
DRAW_LABELS = True               # Рисовать ли текст на проверочных фото

# Создаем структуру папок
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualized", exist_ok=True)

def generate():
    # 1. Инициализация моделей
    print("Инициализация моделей... Подождите.")
    segmentator = seg()
    classifier = SymbolClassifier()
    
    # Подготовка YAML для YOLO (только 1 класс - кубик)
    class_names = classifier.class_names # Должен быть список или dict
    class_to_id = {name: i for i, name in enumerate(class_names)}
    data_yaml = {
        'path': os.path.abspath(OUTPUT_DIR), 
        'train': 'images',
        'val': 'images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(f"{OUTPUT_DIR}/data.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    # Список файлов
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 2. Основной цикл с фиксированным прогресс-баром
    # position=0 и leave=True удерживают полоску внизу
    pbar = tqdm(image_files, desc="Разметка", unit="img", position=0, leave=True)
    
    for filename in pbar:
        img_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        img_viz = image.copy() 


        crops, bboxes = segmentator.get_objects(image)
        
        
        
        
      
        
        yolo_labels = []
        
        
        found_count = 0
        for mask_tensor in crops:
            full_mask = (mask_tensor * 255).astype(np.uint8)
            coords = np.where(full_mask > 0)
            
            if coords[0].size == 0: continue
            
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            
            # # Классификация кропа через CNN
            cropped = image[y1:y2, x1:x2]
            cropped_mask = full_mask[y1:y2, x1:x2]
            input_for_cnn = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)
            
            label_name, confidence = classifier.get_object(input_for_cnn)
            
            if confidence < CONF_THRESHOLD:
                found_count += 1
                # Формат YOLO: class x_center y_center width height (0-1)
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                x_center = (x1 + (x2 - x1) / 2) / w
                y_center = (y1 + (y2 - y1) / 2) / h
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
                
                # Рисуем рамку для визуального контроля
                cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if DRAW_LABELS:
                    txt = f"{label_name} {confidence:.1f}%"
                    cv2.putText(img_viz, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Обновляем инфо в прогресс-баре вместо print()
        pbar.set_postfix({"file": filename, "found": found_count})

        # 3. Сохранение
        if yolo_labels:
            # Картинка для обучения
            cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}", image)
            # Разметка
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(f"{OUTPUT_DIR}/labels/{txt_name}", 'w') as f:
                f.write("\n".join(yolo_labels))
            # Фото для проверки глазами
            cv2.imwrite(f"{OUTPUT_DIR}/visualized/{filename}", img_viz)
        else:
            # Если ничего не нашли или уверенность низкая — выводим лог, не ломая pbar
            tqdm.write(f" [!] Пропущено: {filename} (низкая уверенность или объект не найден)")

    print(f"\nГотово! Датасет собран в папке: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate()