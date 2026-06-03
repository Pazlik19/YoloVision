import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from Coardinate import get_real_coords
from bufer import Scene, Letter

# --- Настройка путей и шрифта ---
MODEL_DIR = Path(__file__).parent.parent / "Models"
# Путь к твоей НОВОЙ модели, которую мы обучили (например, из папки weights/best.pt)
NEW_YOLO_MODEL = MODEL_DIR / 'bestYoloClass3.pt' 
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 

sc = Scene()

def draw_russian_text(image, text, position, font_path, font_size=24, color=(0, 255, 0)):
    """Отрисовка текста с поддержкой кириллицы через PIL"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- Инициализация одной модели ---
# Загружаем YOLO, которая делает и детекцию, и классификацию
model = YOLO(NEW_YOLO_MODEL)
model.to('cuda') 

cap = cv2.VideoCapture(0)
print(f"Запуск на одной модели YOLO... Нажмите 'q' для выхода.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Запуск инференса
    # Мы получаем и координаты, и классы за один проход
    results = model(frame, stream=0, conf=0.8, device='cuda')

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Получаем координаты рамки
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Получаем ID класса и уверенность напрямую из YOLO
            class_id = int(box.cls[0])
            label = model.names[class_id]  # Имя класса (буква)
            confidence = float(box.conf[0]) * 100 # В процентах

            # Расчет реальных координат
            xc, yc = (x2 + x1) / 2, (y2 + y1) / 2
            xr, yr = get_real_coords(xc, yc, 1)

            # Проверка границ
            if x1 < 0 or y1 < 0: continue

            # 2. ОБРАБОТКА И ВИЗУАЛИЗАЦИЯ
            # Добавляем в твой буфер сцены
            sc.handle_detection(x=xr, y=yr, angle=0, label=label)

            # Рисуем рамку
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Пишем текст (теперь label берется сразу из YOLO)
            text = f"{label} ({confidence:.1f}%)"
            frame = draw_russian_text(frame, text, (x1, y1 - 35), FONT_PATH)
            
            print(f"YOLO нашла: {label} ({confidence:.1f}%) в координатах {xr}, {yr}")

    # Показываем результат
    cv2.imshow("Single YOLO Pipeline (Real-time)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()