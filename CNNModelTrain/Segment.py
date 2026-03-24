import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm # Библиотека для индикатора прогресса
from pathlib import Path

def crop_objects_from_directory(input_dir, output_dir, model_path='yolov8n.pt'):
    # 1. Загружаем модель YOLO
    model = YOLO(model_path)
    
    # 2. Подготовка директорий
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Допустимые расширения изображений
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Получаем список файлов
    all_files = os.listdir(input_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"Изображения в папке '{input_dir}' не найдены.")
        return

    print(f"Найдено {len(image_files)} изображений. Начинаю обработку...")

    # Основной цикл по изображениям с прогресс-баром
    for filename in tqdm(image_files):
        # Полный путь к исходному файлу
        img_path = os.path.join(input_dir, filename)
        
        # Читаем изображение через OpenCV
        # (Нам нужна матрица пикселей для кроппинга)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Ошибка при загрузке {filename}. Пропускаю.")
            continue

        # 3. Запускаем YOLO детекцию (verbose=False отключает лог в консоли)
        results = model(image, verbose=False)

        for result in results:
            # Словарь названий классов, чтобы использовать имя в названии файла
            names = result.names
            
            # r.boxes содержит всю информацию о рамках
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # 4. Получаем координаты xyxy (левый верхний и правый нижний углы)
                # Конвертируем их в целые числа
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                
                # Получaем ID класса и имя
                cls = int(box.cls[0].item())
                class_name = names[cls]
                
                # Проверка на корректность координат (иногда бывает 0-size crop)
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x1 >= x2 or y1 >= y2:
                    continue

                # 5. КРОППИНГ через NumPy слайсинг [y1:y2, x1:x2]
                crop = image[y1:y2, x1:x2]
                
                # Проверка, что кроп не пуст
                if crop.size == 0:
                    continue

                # 6. Генерируем уникальное имя для кропа
                # Шаблон: оригинальноеИмя_класс_индексВКадре.jpg
                original_name_no_ext = os.path.splitext(filename)[0]
                crop_name = f"{original_name_no_ext}_{class_name}_{i}.jpg"
                crop_path = os.path.join(output_dir, crop_name)
                
                # Сохраняем вырезанный объект
                cv2.imwrite(crop_path, crop)

    print(f"\nГотово! Вырезанные объекты сохранены в папке '{output_dir}'.")

# --- Настройки путей ---
INPUT_IMAGES_DIR = 'Images'    # Папка с исходными фото
OUTPUT_CROPS_DIR = 'detected_objects' # Папка, куда сохранять вырезанные объекты
# Используем yolov8n.pt (она обучена на COCO: люди, машины, животные и т.д.)
# Если ты обучил свою модель (например, на символы), укажи путь к ней:
# MODEL_WEIGHTS = 'weights/best.pt'
CURENT_DIR = Path(__file__).parent
MODEL_WEIGHTS = CURENT_DIR.parent / "Models"/ "best.pt"

if __name__ == "__main__":
    crop_objects_from_directory(INPUT_IMAGES_DIR, OUTPUT_CROPS_DIR, MODEL_WEIGHTS)