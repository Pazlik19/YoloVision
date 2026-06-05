import cv2
import os
import torch
import json
import numpy as np
from tqdm import tqdm
import yaml
from ultralytics.models.sam import SAM3SemanticPredictor

# --- НАСТРОЙКИ ---
INPUT_DIR = 'Data'                     # Папка с исходными фото
OUTPUT_DIR = 'yolo_dataset_sam3'       # Новое имя корневой папки для датасета
CONF_THRESHOLD = 85.0                  # Порог уверенности (для фильтрации)
DRAW_LABELS = True                     # Рисовать ли текст на проверочных фото

# Автоматически создаем требуемую структуру папок
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualized", exist_ok=True)


def initialization_segmentator():
    """Инициализация модели SAM 3 из встроенного класса Ultralytics"""
    return SAM3SemanticPredictor(overrides=dict(
        task="segment", 
        mode="predict", 
        model="sam3.pt",
        imgsz=644,
        verbose=False,  # Отключает лишний спам в консоль от модели
        save=False      # Отключает автоматическое сохранение результатов в runs/
    ))


def generate():
    # 1. Инициализация модели
    print("Инициализация моделей... Подождите.")
    segmentator = initialization_segmentator()
    
    # Подготовка YAML для YOLO (1 класс - кубик)
    class_names_yolo = {0: "cube"}
    data_yaml = {
        'path': os.path.abspath(OUTPUT_DIR), 
        'train': 'images',
        'val': 'images',
        'names': class_names_yolo
    }
    
    # Сохраняем data.yaml прямо в корень папки yolo_dataset_sam3
    with open(f"{OUTPUT_DIR}/data.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    # Проверяем наличие папки и картинок
    if not os.path.exists(INPUT_DIR):
        print(f" [!] Ошибка: Папка '{INPUT_DIR}' не найдена. Создайте её и положите туда фото.")
        return

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f" [!] В папке '{INPUT_DIR}' не найдено изображений.")
        return
    
    # 2. Основной цикл разметки
    pbar = tqdm(image_files, desc="Разметка", unit="img", position=0, leave=True)
    
    for filename in pbar:
        img_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        img_viz = image.copy() 
        
        # Передаем картинку в SAM и ищем кубик
        segmentator.set_image(image)
        results = segmentator(text=["the cube"])
        
        if not results[0].masks:
            tqdm.write(f" [!] Пропущено: {filename} (Объект не обнаружен нейросетью)")
            continue
        
        yolo_labels = []
        masks = results[0].masks.data.cpu().numpy()
        
        found_count = 0
        for mask_tensor in masks:
            full_mask = (mask_tensor * 255).astype(np.uint8)
            coords = np.where(full_mask > 0)
            
            if coords[0].size == 0: 
                continue
            
            # Координаты описывающего прямоугольника (Bounding Box)
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            
            found_count += 1
            
            # Расчет формата YOLO: class x_center y_center width height (все значения от 0 до 1)
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            x_center = (x1 + (x2 - x1) / 2) / w
            y_center = (y1 + (y2 - y1) / 2) / h
            
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            
            # Отрисовка рамки зеленого цвета для визуального контроля
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Обновляем информацию прямо в строке прогресс-бара
        pbar.set_postfix({"file": filename, "found": found_count})

        # 3. Сохранение результатов в новую структуру
        if yolo_labels:
            # Оригинальное фото в yolo_dataset_sam3/images/
            cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}", image)
            
            # Текстовый файл разметки в yolo_dataset_sam3/labels/
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(f"{OUTPUT_DIR}/labels/{txt_name}", 'w') as f:
                f.write("\n".join(yolo_labels))
                
            # Фото с нарисованной рамкой в yolo_dataset_sam3/visualized/
            cv2.imwrite(f"{OUTPUT_DIR}/visualized/{filename}", img_viz)

    print(f"\nГотово! Датасет успешно собран в папке: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()