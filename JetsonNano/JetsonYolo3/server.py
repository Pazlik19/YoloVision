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

            t_decode = time.time()  # конец JPEG-декода входящего кадра

            height, width = frame.shape[:2]
            texts_to_draw = []
            frame_data = [] # МАССИВ ДЛЯ ДАННЫХ ОБЪЕКТОВ (Телеметрический канал)
            angle_ms = 0.0      # аккумулятор времени get_letter_angle по всем объектам
            classify_ms = 0.0   # время predict_batch (препроцессинг + forward)

            # Выполнение инференса детектирующей нейросети
            results = detector(frame, stream=False, conf=0.6, device=device, verbose=False)
            num_objects = len(results[0].boxes) if results else 0
            t_detect = time.time()  # конец YOLO-детекции

            for r in results:
                # ВАЖНО: Убираем r.plot(), так как теперь мы рисуем рамки вручную
                boxes = r.boxes

                # --- Проход 1: геометрия боксов и сбор кропов ---
                # Сначала собираем все валидные объекты, чтобы классифицировать их
                # ОДНИМ батчем (см. проход 2), а не по одному в цикле.
                dets = []
                for i, box in enumerate(boxes):
                    # Валидация и фильтрация координат Bounding Box
                    x1, y1, x2, y2 = max(0, int(box.xyxy[0][0])), max(0, int(box.xyxy[0][1])), min(width, int(box.xyxy[0][2])), min(height, int(box.xyxy[0][3]))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Вычисление геометрического центра и пересчет в метрические координаты робота
                    xc, yc = (x2 + x1) / 2, (y2 + y1) / 2
                    # Передаём фактический размер кадра (width, height), чтобы оптический
                    # центр и GSD считались от реального разрешения, а не от зашитого 1280x720.
                    xr, yr = get_real_coords(xc, yc, CAMERA_HEIGHT_MM, width, height)

                    dets.append({
                        "i": i,
                        "box": (x1, y1, x2, y2),
                        "crop": frame[y1:y2, x1:x2],
                        "xr": xr,
                        "yr": yr,
                        "box_obj": box,
                    })

                # --- Пакетная классификация: один forward на все кропы ---
                # В режиме yolo_only метки берутся напрямую из YOLO, классификатор не нужен.
                if args.mode == "yolo_resnet":
                    _t = time.time()
                    predictions = classifier.predict_batch([d["crop"] for d in dets])
                    classify_ms += (time.time() - _t) * 1000
                else:
                    predictions = [None] * len(dets)

                # --- Проход 2: классы, угол, отрисовка, телеметрия ---
                for d, pred in zip(dets, predictions):
                    i = d["i"]
                    x1, y1, x2, y2 = d["box"]
                    crop = d["crop"]
                    xr, yr = d["xr"], d["yr"]

                    engel = 0
                    display_label = "Unknown"
                    confidence = 0.0

                    # По умолчанию считаем, что объект нашел только YOLO (Синяя рамка)
                    box_color = COLOR_YOLO_ONLY

                    # Сценарий детекции YOLO + классификации ResNet
                    if args.mode == "yolo_resnet":
                        label, confidence = pred

                        if confidence < 60:
                            display_label = "Low conf"
                            # Цвет остается синим
                        else:
                            display_label = label
                            box_color = COLOR_CLASSIFIED # Успешная классификация ResNet — Зеленая рамка
                            _t = time.time()
                            engel = get_letter_angle(crop)  # полный угол буквы (0..360), откат на угол грани
                            angle_ms += (time.time() - _t) * 1000

                    # Сценарий сквозной детекции YOLO
                    elif args.mode == "yolo_only":
                        class_id = int(d["box_obj"].cls[0])
                        display_label = r.names[class_id]
                        confidence = float(d["box_obj"].conf[0]) * 100
                        box_color = COLOR_CLASSIFIED # В этом режиме YOLO делает всё, ставим зеленую
                        _t = time.time()
                        engel = get_letter_angle(crop)  # полный угол буквы (0..360)
                        angle_ms += (time.time() - _t) * 1000

                    # --- ОТРИСОВКА РАМКИ (Bounding Box) ---
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Формируем структуру данных текущего объекта для отправки
                    frame_data.append({
                        "label": display_label,
                        "x": xr,
                        "y": yr,
                        "angle": engel
                    })

                    # Компактная подпись над/под боксом: буква, уверенность, угол в одну
                    # строку. Префикс "latter_" убираем. Чётные боксы подписываем сверху,
                    # нечётные — снизу, чтобы соседние подписи в плотном ряду не наезжали.
                    short = display_label.split("_")[-1]
                    caption = f"{i}:{short} {confidence:.0f}%"
                    if engel:
                        caption += f" {engel:.0f}°"
                    ty = max(0, y1 - 18) if i % 2 == 0 else min(height - 16, y2 + 2)
                    texts_to_draw.append((caption, (x1, ty)))

            t_loop = time.time()  # конец цикла по объектам (классификация + угол + геометрия)

            # Рендеринг кириллических и латинских символов средствами PIL
            if texts_to_draw:
                frame = draw_multiple_texts(frame, texts_to_draw, DEFAULT_FONT_PATH, font_size=15)
            t_draw = time.time()

            # Подготовка Multipart-пакета: сжатый кадр + JSON-телеметрия
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            json_telemetry = json.dumps(frame_data)
            t_encode = time.time()

            # Синхронная отправка сообщения из двух кадров
            socket.send_multipart([
                buffer.tobytes(),
                json_telemetry.encode('utf-8')
            ])

            # --- Инструментация: разбивка времени кадра по стадиям ---
            # decode  — JPEG-декод входящего кадра
            # detect  — инференс YOLO (TRT)
            # classify— predict_batch: prep (PIL-препроцессинг, CPU) + fwd (TRT-forward)
            # angle   — суммарно get_letter_angle по всем объектам (CPU)
            # misc    — остальное в цикле (геометрия, рамки, формирование подписей)
            # draw    — наложение текста через PIL
            # encode  — JPEG-кодирование исходящего кадра
            pt = classifier.last_timings if classifier is not None else {"preprocess_ms": 0.0, "forward_ms": 0.0}
            total = (time.time() - start_time) * 1000
            misc_ms = (t_loop - t_detect) * 1000 - classify_ms - angle_ms
            print(
                f"[{args.mode}] total={total:.0f}ms | "
                f"decode={(t_decode - start_time) * 1000:.0f} "
                f"detect={(t_detect - t_decode) * 1000:.0f} "
                f"classify={classify_ms:.0f}(prep={pt['preprocess_ms']:.0f}/fwd={pt['forward_ms']:.0f}) "
                f"angle={angle_ms:.0f} "
                f"misc={misc_ms:.0f} "
                f"draw={(t_draw - t_loop) * 1000:.0f} "
                f"encode={(t_encode - t_draw) * 1000:.0f} | obj={num_objects}"
            )

    except KeyboardInterrupt:
        print("\n[INFO] Сервер остановлен пользователем (KeyboardInterrupt).")
    finally:
        # Корректное закрытие ресурсов сетевой топологии
        socket.close()
        context.term()
        print("[INFO] Ресурсы сокетов ZeroMQ успешно деинициализированы.")

if __name__ == '__main__':
    main()