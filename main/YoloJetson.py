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
try:
    detector = YOLO(MODEL_DIR / 'best.pt') 
    classifier = SymbolClassifier()
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")
    sys.exit(1)

# Инициализация камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Камера не найдена!")
    sys.exit(1)

# --- АВТООПРЕДЕЛЕНИЕ РАЗМЕРА КАДРА ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Камера определена: {width}x{height}")

# Настройка записи (теперь с правильным размером)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_vision.avi', fourcc, 10.0, (width, height))

print("Запись пошла! Для остановки нажми Ctrl+C в терминале.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO детекция
        results = detector(frame, stream=False, conf=0.6, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Кроп с защитой границ
                crop = frame[max(0, y1):min(height, y2), 
                             max(0, x1):min(width, x2)]
                
                if crop.size > 0:
                    try:
                        label, confidence = classifier.predict(crop)
                        if confidence > 70: # Снизил порог для теста
                            # Рисуем рамку и текст в кадр
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {confidence:.0f}%", (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            print(f"Детекция: {label} ({confidence:.0f}%)")
                    except Exception as e:
                        pass

        # Записываем кадр (теперь размер совпадает на 100%)
        out.write(frame)

except KeyboardInterrupt:
    print("\nОстановка записи...")

finally:
    cap.release()
    out.release()
    # Убрали destroyAllWindows, чтобы не было ошибок "not implemented"
    print(f"Готово! Файл сохранен: {Path.cwd()}/output_vision.avi")