import cv2
import zmq
import numpy as np
import sys
import time
import json
import threading
from flask import Flask, Response, jsonify
import random

# Импортируем буфер сцены (файл bufer.py должен лежать в той же директории)
from bufer import Scene 

# --- Конфигурация системы ---
# Если Docker запущен с флагом --net=host, используем localhost. 
# Иначе замените на IP-адрес контейнера.
ZMQ_ADDRESS = "tcp://localhost:5555" 

app = Flask(__name__)

# Глобальные разделяемые переменные между потоками
output_frame = None
frame_lock = threading.Lock()  # Обеспечивает потокобезопасность при копировании кадров
scene_buffer = Scene()         # Инициализация буфера накопления объектов




class MockOdometry:
    """Генератор случайных координат камеры для тестирования глобальной карты"""
    def __init__(self, interval_seconds=30):
        self.interval = interval_seconds
        self.last_update = time.time()
        
        # Стартовые координаты камеры (в мм)
        self.cam_x = 0.0
        self.cam_y = 0.0

    def get_position(self):
        current_time = time.time()
        if current_time - self.last_update >= self.interval:
            # Смещаем камеру на случайное расстояние (например, от -500 до +500 мм)
            self.cam_x += random.uniform(-500.0, 500.0)
            self.cam_y += random.uniform(-500.0, 500.0)
            self.last_update = current_time
            print(f"\n[MOCK ODOMETRY] Камера переместилась! Новые координаты: X={self.cam_x:.1f}, Y={self.cam_y:.1f}\n")
            
        return self.cam_x, self.cam_y

# Инициализируем мок-модуль
camera_tracker = MockOdometry(interval_seconds=30)

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, 
                       display_width=640, display_height=480, framerate=30, flip_method=0):
    """Спецификация конвейера GStreamer для работы с аппаратным ускорителем NVArgus."""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

# --- Логика веб-интерфейса Flask ---
def generate():
    """Генератор MJPEG-потока для трансляции в веб-браузер."""
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            # Кодируем текущий обработанный кадр обратно в JPG для HTTP-отдачи
            ret, encoded_image = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_bytes = encoded_image.tobytes()
        
        # Формируем multipart HTTP-ответ (кадр за кадром)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Ограничение частоты отдачи (~30 FPS)

@app.route("/")
def video_feed():
    """Главная страница: видеопоток MJPEG с наложенной графикой аналитики."""
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def get_scene_data():
    """Телеметрический эндпоинт: отдает накопленные в буфере данные в JSON."""
    return jsonify({
        "current_word": scene_buffer.get_word(),
        "objects_count": len(scene_buffer.objects),
        "timestamp": time.time(),
        "objects": scene_buffer.get_all_objects_data()  
    })

# --- Основной вычислительный поток (Захват + ZMQ) ---
def run_vision():
    global output_frame, scene_buffer
    
    # Настройка REQ-сокета для отправки запросов в сторону Docker
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(ZMQ_ADDRESS)
    
    # Инициализация CSI-камеры через GStreamer
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    print(f"[INFO] Поток захвата запущен. Подключение к ZMQ: {ZMQ_ADDRESS}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("[ERROR] Не удалось получить кадр с CSI-камеры")
                break
            
            loop_start = time.time()
            
            # Сжимаем исходный кадр в JPEG перед отправкой, чтобы разгрузить шину
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            socket.send(buffer.tobytes())
            
            # Принимаем multipart-ответ: [0] - Изображение, [1] - Телеметрия JSON
            reply = socket.recv_multipart()
            
            if len(reply) == 1 and reply[0] == b"ERROR": 
                print("[WARNING] Сервер детекции вернул ошибку")
                continue
                
            image_bytes = reply[0]
            json_bytes = reply[1]
            
            # Декодируем размеченный кадр, пришедший из Docker
            np_frame = np.frombuffer(image_bytes, dtype=np.uint8)
            processed = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            # Десериализуем данные об объектах и передаем в локальный буфер хоста
            telemetry = json.loads(json_bytes.decode('utf-8'))
            
            valid_detections = []
            for obj in telemetry:
                if obj["label"] not in ["Unknown", "Low conf"]:
                    valid_detections.append(obj)
                    
            # 1. Получаем текущие координаты камеры из нашего мок-генератора
            current_cam_x, current_cam_y = camera_tracker.get_position()
            
            # 2. Обновляем сцену, передавая массив объектов и РЕАЛЬНУЮ позицию камеры.
            #    Без этого все объекты складываются в координатах центра кадра, а буфер
            #    считает, что камера всегда в (0,0), и удаляет уехавшие из вида объекты.
            scene_buffer.update_scene(
                current_detections=valid_detections,
                cam_x=current_cam_x,
                cam_y=current_cam_y
            )


            # Расчет полного времени цикла (RTT задержка)
            total_time = (time.time() - loop_start) * 1000
            cv2.putText(processed, f"Latency: {total_time:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Потокобезопасно обновляем глобальный кадр для Flask
            with frame_lock:
                output_frame = processed.copy()

    except Exception as e:
        print(f"[CRITICAL] Ошибка в цикле обработки: {e}")
    finally:
        cap.release()
        socket.close()
        context.term()
        print("[INFO] Ресурсы камеры и сокетов на хосте успешно освобождены.")

if __name__ == '__main__':
    # 1. Запускаем фоновый поток для работы с камерой и ZMQ, чтобы он не блокировал Flask
    vision_thread = threading.Thread(target=run_vision, daemon=True)
    vision_thread.start()
    
    print("\n" + "="*50)
    print("Система успешно запущена на Jetson Nano!")
    print("Стрим видео (MJPEG):  http://0.0.0.0:5000")
    print("Данные буфера (JSON): http://0.0.0.0:5000/data")
    print("="*50 + "\n")
    
    # 2. Запускаем веб-сервер Flask в главном потоке
    # threaded=True позволяет обрабатывать запросы к видео и к эндпоинту /data параллельно
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)