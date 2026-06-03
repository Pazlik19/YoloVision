import cv2
import os
import time
import threading
import paramiko
from flask import Flask, Response

# ==================== НАСТРОЙКИ СИСТЕМЫ ====================
SAVE_PERIOD = 2.0               # Интервал сохранения (1.0 = раз в секунду)
LOCAL_DIR = "./dataset_raw"     # Локальная папка на Jetson для бэкапа

# Настройки вашего основного компьютера (куда отправлять файлы)
USE_SSH = True                  # Переключите в False, если хотите копить только локально
SSH_HOST = "10.185.86.189"      # IP-адрес вашего ПК
SSH_PORT = 22
SSH_USER = "jetson"
SSH_PASSWORD = "Jetson_123"
SSH_REMOTE_DIR = "/C:/dataset_raw" # Папка на ПК (должна быть создана заранее)
# ===========================================================

app = Flask(__name__)

# Глобальные переменные для обмена кадрами между потоками
latest_frame = None
frame_lock = threading.Lock()

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, 
                       display_width=640, display_height=480, framerate=30, flip_method=0):
    """Стандартный конвейер GStreamer для работы с CSI-камерой на Jetson Nano."""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

def camera_worker():
    """Поток непрерывного захвата кадров с камеры."""
    global latest_frame
    
    pipeline = gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[ОШИБКА] Не удалось запустить CSI-камеру через GStreamer!")
        return

    print("[КАМЕРА] Захват видео успешно запущен.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Потокобезопасно обновляем текущий кадр
        with frame_lock:
            latest_frame = frame.copy()
            
    cap.release()

def dataset_worker():
    """Поток сбора датасета: раз в секунду сохраняет и шлет кадр через удерживаемое SSH соединение."""
    print(f"[ДАТАСЕТ] Фоновый поток запущен. Интервал: {SAVE_PERIOD} сек.")
    
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
        
    ssh_client = None
    sftp = None

    while True:
        time.sleep(SAVE_PERIOD)
        
        # 1. Забираем текущий чистый кадр
        with frame_lock:
            if latest_frame is None:
                continue
            frame_to_save = latest_frame.copy()
            
        # Формируем имя файла (Дата_Время_Миллисекунды)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"img_{timestamp}_{ms:03d}.jpg"
        local_path = os.path.join(LOCAL_DIR, filename)
        
        try:
            # 2. Сохраняем локально на Jetson
            cv2.imwrite(local_path, frame_to_save)
            
            if not USE_SSH:
                print(f"[ЛОКАЛЬНО] Сохранен кадр: {filename}")
                continue
                
            # 3. Поддерживаем открытым постоянный SSH/SFTP канал
            if ssh_client is None or not ssh_client.get_transport() or not ssh_client.get_transport().is_active():
                print("[SSH] Подключение к компьютеру...")
                try:
                    ssh_client = paramiko.SSHClient()
                    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=5)
                    sftp = ssh_client.open_sftp()
                    print("[SSH] Успешно подключено. Канал отправки открыт.")
                except Exception as conn_err:
                    print(f"[SSH ОШИБКА СВЯЗИ] Не удалось подключиться к ПК: {conn_err}")
                    ssh_client = None
                    continue # Кадр остался локально, попробуем отправить следующий через секунду

            # 4. Быстрая отправка в открытый канал
            remote_path = os.path.join(SSH_REMOTE_DIR, filename).replace("\\", "/")
            sftp.put(local_path, remote_path)
            print(f"[ДАТАСЕТ] Файл {filename} успешно улетел на ПК.")
            
        except Exception as e:
            print(f"[ОШИБКА ОТПРАВКИ] Сбой на файле {filename}: {e}")
            # Сбрасываем сессию, чтобы на следующем цикле скрипт переподключился к SSH автоматически
            sftp = None
            ssh_client = None

def generate_mjpeg_frames():
    """Генератор кадров MJPEG для Flask-трансляции."""
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            # Сжимаем для вывода в веб (качество 75%, чтобы не нагружать сеть Jetson)
            _, buffer = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # Ограничиваем стрим в браузере до ~25 FPS

@app.route('/')
def video_feed():
    """Эндпоинт для просмотра видеопотока в браузере."""
    return Response(generate_mjpeg_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Поток 1: Постоянный захват камеры
    t_camera = threading.Thread(target=camera_worker, daemon=True)
    t_camera.start()
    
    # Поток 2: Таймер датасета (1 кадр в секунду + SSH)
    t_dataset = threading.Thread(target=dataset_worker, daemon=True)
    t_dataset.start()
    
    print("\n" + "="*60)
    print(" ЧИСТЫЙ СКРИПТ СБОРА ДАТАСЕТА ЗАПУЩЕН")
    print(f" Локальный бэкап: {LOCAL_DIR}")
    print(f" Стрим для контроля камеры: http://<IP_JETSON>:5000/")
    print("="*60 + "\n")
    
    # Запуск веб-сервера Flask в основном потоке
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)