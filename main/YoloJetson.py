import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from TextDetection2 import SymbolClassifier
from pathlib import Path
import sys

# Определение путей
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "Models"

# --- Инициализация моделей ---
print("Загрузка моделей...")
try:
    # 1. Детектор (YOLO)
    detector = YOLO(MODEL_DIR / 'best.pt') 

    # 2. Твой классификатор (ResNet)
    classifier = SymbolClassifier()
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")
    sys.exit(1)

# --- Настройка камеры и записи видео ---
cap = cv2.VideoCapture(0)

# Настройки для сохранения видео (VideoWriter)
# Используем кодек XVID, он хорошо работает на Linux/Jetson
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0  # Частота кадров (можно подстроить под реальную скорость Jetson)
frame_size = (640, 480) # Стандарт для большинства камер, проверь если у тебя другое
out = cv2.VideoWriter('output_vision.avi', fourcc, fps, frame_size)

print("Запуск захвата... Для остановки нажмите Ctrl+C в терминале.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        # 1. YOLO ищет объекты (кубики)
        # conf=0.9 — очень высокий порог, если не будет находить, снизь до 0.5
        results = detector(frame, stream=False, conf=0.9, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Получаем координаты бокса
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Защита от выхода за границы кадра
                if x1 < 0 or y1 < 0: continue

                # 2. КРОППИНГ
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue

                # 3. КЛАССИФИКАЦИЯ (ResNet)
                try:
                    label, confidence = classifier.predict(crop)
                    
                    # Если уверенность ResNet низкая, пропускаем
                    if confidence < 90:
                        continue
                        
                    # 4. ВИЗУАЛИЗАЦИЯ (рисуем прямо в кадре перед записью)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Логируем в консоль для отладки
                    print(f"Обнаружено: {label} ({confidence:.1f}%)")
                    
                    text = f"{label} ({confidence:.1f}%)"
                    cv2.putText(frame, text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Ошибка классификации: {e}")

        # --- ЗАПИСЬ КАДРА В ФАЙЛ ---
        # Важно: размер кадра в frame должен совпадать с frame_size в VideoWriter
        out.write(frame)

        # В Docker imshow не сработает, поэтому мы его закомментировали
        # cv2.imshow("YOLO + ResNet Classification", frame)

        # Оставляем waitKey для корректной работы OpenCV
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nОстановка записи пользователем...")

finally:
    # --- КРИТИЧНО: Закрываем всё правильно ---
    cap.release()
    out.release() # Без этого файл видео будет битым!
    cv2.destroyAllWindows()
    print("Программа завершена. Видео сохранено в 'output_vision.avi'.")