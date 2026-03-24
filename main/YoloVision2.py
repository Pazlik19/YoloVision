import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from TextDetection2 import SymbolClassifier
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "Models"
# Импортируем твой класс (убедись, что он в этом же файле или импортирован корректно)
# class SymbolClassifier: ... (твой код из примера выше)

# --- Инициализация моделей ---
# 1. Детектор (YOLO) - находит объекты
detector = YOLO(MODEL_DIR/'best.pt') 

# 2. Твой классификатор - уточняет, что это
# Убедись, что файлы 'best_resnet_model.pth' и 'class_names.json' лежат рядом
classifier = SymbolClassifier()

# --- Работа с камерой ---
cap = cv2.VideoCapture(0)

print("Запуск... Нажмите 'q' для выхода.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO ищет объекты
    results = detector(frame, stream=0,conf=0.9)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Получаем координаты бокса (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Проверка на корректность координат (чтобы не вырезать пустой кадр)
            if x1 < 0 or y1 < 0: continue

            # 2. КРОППИНГ: Вырезаем объект из кадра
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue

            # 3. КЛАССИФИКАЦИЯ: Отправляем вырезанный кусок в твой ResNet
            label, confidence = classifier.predict(crop)
            if confidence < 90:
                continue
            # 4. ВИЗУАЛИЗАЦИЯ
            # Рисуем рамку
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(label)
            # Пишем текст (Класс и Уверенность из твоего SymbolClassifier)
            text = f"{label} ({confidence:.1f}%)"
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Показываем результат
    cv2.imshow("YOLO + ResNet Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()