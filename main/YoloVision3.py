import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from TextDetection2 import SymbolClassifier
from pathlib import Path
from Coardinate import get_real_coords

# --- Настройка путей и шрифта ---
MODEL_DIR = Path(__file__).parent.parent / "Models"
# Стандартный путь к шрифту в Windows. Если у вас его нет, замените на любой .ttf
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 

def draw_russian_text(image, text, position, font_path, font_size=24, color=(0, 255, 0)):
    """Функция для отрисовки текста с поддержкой кириллицы"""
    # Конвертируем BGR (OpenCV) -> RGB (PIL)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    # Конвертируем обратно RGB -> BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- Инициализация моделей ---
# 1. Детектор (YOLO) - отправляем на CUDA
detector = YOLO(MODEL_DIR/'best.pt') 
detector.to('cuda') 

# 2. Классификатор (убедитесь, что внутри него тоже используется .to('cuda'))
classifier = SymbolClassifier()

cap = cv2.VideoCapture(0)
print("Запуск... Нажмите 'q' для выхода.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO ищет объекты (используем device='cuda' для надежности)
    results = detector(frame, stream=0, conf=0.9, device='cuda')

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            xc , yc = (x2+x1)/2 , (y2 + y1)/2
            xr , yr = get_real_coords(xc,yc, 1)
            if x1 < 0 or y1 < 0: continue

            # 2. КРОППИНГ
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 3. КЛАССИФИКАЦИЯ
            label, confidence = classifier.predict(crop)



            
            if confidence < 90:
                continue

            # 4. ВИЗУАЛИЗАЦИЯ
            # Рисуем рамку средствами OpenCV
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Рисуем русский текст средствами PIL через нашу функцию
            text = f"{label} ({confidence:.1f}%)"
            frame = draw_russian_text(frame, text, (x1, y1 - 35), FONT_PATH)
            
            print(f"Обнаружено: {label} с уверенностью {confidence:.1f}%")

    # Показываем результат
    cv2.imshow("YOLO + ResNet (CUDA Accelerated)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()