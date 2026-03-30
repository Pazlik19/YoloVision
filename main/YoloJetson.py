import cv2
import torch
from ultralytics import YOLO
from TextDetection2 import SymbolClassifier
from pathlib import Path
import sys

# Пути
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "Models"

# Загрузка моделей
print("Загрузка моделей...")
detector = YOLO(MODEL_DIR / 'best.pt') 
classifier = SymbolClassifier()

# Камера
cap = cv2.VideoCapture(0)

# Настройка записи (VideoWriter)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_vision.avi', fourcc, 10.0, (640, 480))

print("Запись пошла! Крути кубики. Для остановки нажми Ctrl+C в терминале.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO детекция
        results = detector(frame, stream=False, conf=0.7, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[max(0, y1):min(frame.shape[0], y2), 
                             max(0, x1):min(frame.shape[1], x2)]
                
                if crop.size > 0:
                    label, confidence = classifier.predict(crop)
                    if confidence > 80:
                        # Рисуем только в файл
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.0f}%", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print(f"Вижу: {label}")

        # Записываем кадр
        out.write(frame)

except KeyboardInterrupt:
    print("\nОстановка...")

finally:
    cap.release()
    out.release()
    print("Готово! Проверяй файл output_vision.avi")