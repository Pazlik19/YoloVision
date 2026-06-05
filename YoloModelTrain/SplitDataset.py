import os
import random
import shutil
import yaml

# --- НАСТРОЙКИ ---
DATASET_DIR = 'yolo_dataset_sam3'  # Имя папки с вашим датасетом
TRAIN_RATIO = 0.8                  # Процент данных для обучения (80%)

def split_dataset():
    src_images_dir = os.path.join(DATASET_DIR, 'images')
    src_labels_dir = os.path.join(DATASET_DIR, 'labels')
    
    # Проверяем, не был ли датасет уже разделен
    if os.path.exists(os.path.join(src_images_dir, 'train')):
        print("[!] Ошибка: Похоже, папка 'train' уже существует. Датасет уже разделен.")
        return

    # Получаем список всех изображений
    all_images = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not all_images:
        print(f"[!] Ошибка: В папке '{src_images_dir}' не найдено картинок для разделения.")
        return

    # Создаем правильную структуру папок для YOLO
    dirs_to_create = [
        os.path.join(src_images_dir, 'train'),
        os.path.join(src_images_dir, 'val'),
        os.path.join(src_labels_dir, 'train'),
        os.path.join(src_labels_dir, 'val')
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    # Перемешиваем файлы случайным образом (seed фиксируем, чтобы результат был воспроизводим)
    random.seed(42)
    random.shuffle(all_images)

    # Считаем индекс для разделения 80/20
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Всего найдено изображений: {len(all_images)}")
    print(f"├── Отправляется в train (80%): {len(train_images)}")
    print(f"└── Отправляется в val (20%): {len(val_images)}")

    # Функция для безопасного перемещения картинок и их txt-файлов разметки
    def move_files(file_list, subset_name):
        for filename in file_list:
            # Перемещаем картинку
            shutil.move(
                os.path.join(src_images_dir, filename), 
                os.path.join(src_images_dir, subset_name, filename)
            )
            
            # Ищем и перемещаем соответствующий txt-файл разметки
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            src_txt_path = os.path.join(src_labels_dir, txt_filename)
            
            if os.path.exists(src_txt_path):
                shutil.move(
                    src_txt_path, 
                    os.path.join(src_labels_dir, subset_name, txt_filename)
                )

    print("\nРазделение файлов...")
    move_files(train_images, 'train')
    move_files(val_images, 'val')

    # Автоматически обновляем структуру внутри data.yaml
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        # Меняем пути на новые подпапки
        data_config['train'] = 'images/train'
        data_config['val'] = 'images/val'

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
        print("[+] Файл data.yaml успешно обновлен.")
    else:
        print("[!] Предупреждение: файл data.yaml не найден в корне датасета.")

    print("\n[+] Успешно! Датасет полностью готов к правильному обучению.")

if __name__ == '__main__':
    split_dataset()