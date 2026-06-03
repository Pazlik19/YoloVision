import cv2
import torch
from ultralytics import YOLO
from TextDetection2 import SymbolClassifier
from pathlib import Path
import sys
import time

# --- Функция GStreamer для CSI-камеры IMX219 ---
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

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

# Инициализация камеры через GStreamer
print("Инициализация GStreamer...")
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру через GStreamer! Проверь подключение шлейфа.")
    sys.exit(1)

# Мы задали display_width/height в конвейере как 640x480
width, height = 640, 480

# Настройка записи (MJPG — самый совместимый формат для Jetson)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_vision.avi', fourcc, 10.0, (width, height))

print("Система готова. Запись пошла! Нажми Ctrl+C для выхода.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Пустой кадр!")
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
                        if confidence > 75:
                            # Рисуем рамку
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Пишем текст
                            text = f"{label} {confidence:.0f}%"
                            cv2.putText(frame, text, (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            print(f"Обнаружен символ: {label} ({confidence:.0f}%)")
                    except Exception as e:
                        pass

        # Записываем обработанный кадр в файл
        out.write(frame)

except KeyboardInterrupt:
    print("\nЗавершение работы...")

finally:
    cap.release()
    out.release()