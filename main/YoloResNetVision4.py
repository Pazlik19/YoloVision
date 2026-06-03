import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from Coardinate import get_real_coords
from bufer import Scene

# --- Настройка путей ---
MODEL_DIR = Path(__file__).parent.parent / "Models"
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 

sc = Scene()

def draw_all_labels(image, detections, font_path, font_size=24):
    """
    Накладывает сразу все подписи на кадр. 
    detections: список кортежей (x1, y1, text)
    """
    if not detections:
        return image
        
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    for x1, y1, text in detections:
        # Рисуем подложку под текст для читаемости
        draw.text((x1, y1 - 35), text, font=font, fill=(0, 255, 0))
        
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- Инициализация модели ---
# Используем твою новую сбалансированную модель
detector = YOLO(MODEL_DIR / 'YoloClass.pt') 
detector.to('cuda') 

cap = cv2.VideoCapture(0)
print("Запуск YOLO-only режима... Нажмите 'q' для выхода.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Детекция (YOLO сразу классифицирует буквы)
    results = detector(frame, stream=0, conf=0.90, device='cuda'
                       )
    
    labels_to_draw = []

    for r in results:
        boxes = r.boxes
        names = r.names # Словарь имен классов из data.yaml
        
        for box in boxes:
            # Координаты
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names[cls_id] # Получаем букву (кириллицу) по ID
            
            # Координаты центра для логики робота
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            xr, yr = get_real_coords(xc, yc, 1)

            # 2. Фильтрация и логика
            if conf < 0.85: # Наш порог уверенности
                continue

            # Добавляем в буфер сцены
            sc.handle_detection(x=xr, y=yr, angle=0, label=label)
            
            # Рисуем рамку (OpenCV работает быстро)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Собираем данные для отрисовки текста
            text_str = f"{label} {conf:.0%}"
            labels_to_draw.append((x1, y1, text_str))

    # 3. Отрисовка всех надписей за один проход
    frame = draw_all_labels(frame, labels_to_draw, FONT_PATH)

    # Вывод FPS или инфо
    cv2.putText(frame, f"Objects: {len(labels_to_draw)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLO Detection (End-to-End)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()