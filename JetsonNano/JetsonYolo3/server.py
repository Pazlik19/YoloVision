import argparse
import zmq
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import cv2
import torch
import time
import json
from ultralytics import YOLO

# Импорт конфигурационных параметров и утилит проекта
from config import MODEL_DIR, ZMQ_ADDRESS, CAMERA_HEIGHT_MM, DEFAULT_FONT_PATH
from draw_utils import draw_multiple_texts
from Coardinate import get_real_coords, get_angel, get_letter_angle
from TextDetection2 import SymbolClassifier

def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="ZMQ-сервер детекции и классификации объектов для Jetson Nano")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['yolo_resnet', 'yolo_only'], 
        default='yolo_resnet',
        help="Режим работы бэкенда: 'yolo_resnet' (двухэтапный) или 'yolo_only' (сквозной)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"--- Инициализация Сервера в Docker ---")
    print(f"Выбранный режим: {args.mode}")
    print(f"Вычислительное устройство: {device}")

    # Грузим ENGINE вместо PT для оптимизации инференса через TensorRT
    if args.mode == "yolo_resnet":
        detector = YOLO(MODEL_DIR / 'best.engine', task='detect')
        classifier = SymbolClassifier() 
    else:
        detector = YOLO(MODEL_DIR / 'bestYoloClass3.pt', task='detect').to(device)
        classifier = None

    # Настройка контекста и сокета ZeroMQ (Архитектура сокетов REP-REQ)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(ZMQ_ADDRESS)

    print(f"Сервер готов на {ZMQ_ADDRESS}. Ожидание кадров...")

    # Цвета в формате BGR (OpenCV)
    COLOR_YOLO_ONLY = (255, 0, 0)  # Синий
    COLOR_CLASSIFIED = (0, 255, 0) # Зеленый

    try:
        while True:
            # Принимаем байтовый поток изображения от хоста Jetson Nano
            frame_bytes = socket.recv()
            start_time = time.time()
            
            # Восстанавливаем матрицу изображения из буфера памяти
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                socket.send(b"ERROR")
                continue

            height, width = frame.shape[:2]
            texts_to_draw = []
            frame_data = [] # МАССИВ ДЛЯ ДАННЫХ ОБЪЕКТОВ (Телеметрический канал)
            
            # Выполнение инференса детектирующей нейросети
            results = detector(frame, stream=False, conf=0.6, device=device, verbose=False)
            num_objects = len(results[0].boxes) if results else 0

            for r in results:
                # ВАЖНО: Убираем r.plot(), так как теперь мы рисуем рамки вручную
                boxes = r.boxes
                
                for i, box in enumerate(boxes):
                    # Валидация и фильтрация координат Bounding Box
                    x1, y1, x2, y2 = max(0, int(box.xyxy[0][0])), max(0, int(box.xyxy[0][1])), min(width, int(box.xyxy[0][2])), min(height, int(box.xyxy[0][3]))
                    if x2 <= x1 or y2 <= y1: 
                        continue

                    # Вычисление геометрического центра и пересчет в метрические координаты робота
                    xc, yc = (x2 + x1) / 2, (y2 + y1) / 2
                    xr, yr = get_real_coords(xc, yc, CAMERA_HEIGHT_MM)
                    crop = frame[y1:y2, x1:x2]

                    engel = 0
                    display_label = "Unknown"
                    
                    # По умолчанию считаем, что объект нашел только YOLO (Синяя рамка)
                    box_color = COLOR_YOLO_ONLY 

                    # Сценарий детекции YOLO + классификации ResNet
                    if args.mode == "yolo_resnet" and crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        label, confidence = classifier.predict(crop_rgb)
                        
                        if confidence < 60:
                            display_label = "Low conf"
                            # Цвет остается синим
                        else:
                            display_label = label
                            box_color = COLOR_CLASSIFIED # Успешная классификация ResNet — Зеленая рамка
                            engel = get_letter_angle(crop)  # полный угол буквы (0..360), откат на угол грани

                        texts_to_draw.append((f"ID: {i} ({label} {confidence:.1f}%)", (x1, max(0, y1 - 50))))
                    
                    # Сценарий сквозной детекции YOLO
                    elif args.mode == "yolo_only":
                        class_id = int(box.cls[0])
                        display_label = r.names[class_id]
                        confidence = float(box.conf[0]) * 100
                        box_color = COLOR_CLASSIFIED # В этом режиме YOLO делает всё, ставим зеленую
                        
                        if crop.size > 0:
                            engel = get_letter_angle(crop)  # полный угол буквы (0..360)

                        texts_to_draw.append((f"ID: {i} ({display_label} {confidence:.1f}%)", (x1, max(0, y1 - 50))))

                    # --- ОТРИСОВКА РАМКИ (Bounding Box) ---
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Формируем структуру данных текущего объекта для отправки
                    frame_data.append({
                        "label": display_label,
                        "x": xr,
                        "y": yr,
                        "angle": engel
                    })
                    
                    if engel != 0:
                        texts_to_draw.append((f"Angle: {engel} deg", (x1, max(0, y1 - 30))))

            # Рендеринг кириллических и латинских символов средствами PIL
            if texts_to_draw:
                frame = draw_multiple_texts(frame, texts_to_draw, DEFAULT_FONT_PATH)

            # Подготовка Multipart-пакета: сжатый кадр + JSON-телеметрия
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            json_telemetry = json.dumps(frame_data)
            
            # Синхронная отправка сообщения из двух кадров
            socket.send_multipart([
                buffer.tobytes(), 
                json_telemetry.encode('utf-8')
            ])
            
            print(f"[{args.mode}] Замер инференса и отрисовки: {(time.time() - start_time)*1000:.1f}ms. Объектов: {num_objects}")

    except KeyboardInterrupt:
        print("\n[INFO] Сервер остановлен пользователем (KeyboardInterrupt).")
    finally:
        # Корректное закрытие ресурсов сетевой топологии
        socket.close()
        context.term()
        print("[INFO] Ресурсы сокетов ZeroMQ успешно деинициализированы.")

if __name__ == '__main__':
    main()