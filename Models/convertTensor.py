from ultralytics import YOLO
import os

# Путь к вашему весовому файлу PyTorch
model_path = 'best.pt'

if os.path.exists(model_path):
    print("Запуск конвертации best.pt -> best.engine...")
    model = YOLO(model_path)
    # export в TensorRT с оптимизацией FP16 для Jetson
    model.export(format='engine', half=True, device=0)
    print("Экспорт завершен успешно!")
else:
    print(f"Файл {model_path} не найден!")