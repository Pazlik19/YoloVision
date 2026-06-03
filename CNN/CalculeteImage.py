import os
from pathlib import Path

def count_images_in_folders(root_path):
    # Список расширений, которые мы считаем изображениями
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    
    stats = {}
    root = Path(root_path)

    if not root.exists():
        print(f"Ошибка: Путь '{root_path}' не найден.")
        return

    print(f"Анализ папки: {root.absolute()}\n")

    # Проходим по всем подпапкам первого уровня
    for item in root.iterdir():
        if item.is_dir():
            # Считаем файлы с нужными расширениями в текущей подпапке
            count = sum(1 for f in item.iterdir() if f.suffix.lower() in valid_extensions)
            stats[item.name] = count

    if not stats:
        print("Изображения или подпапки не найдены.")
        return

    # Визуализация
    max_label_length = max(len(name) for name in stats.keys())
    max_count = max(stats.values()) if stats.values() else 1
    
    print(f"{'Папка':<{max_label_length}} | {'Кол-во':<6} | График")
    print("-" * (max_label_length + 20))
    count = 0 
    for i, (folder, count) in enumerate(sorted(stats.items())):
        # Рисуем полоску
        bar_length = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "█" * bar_length
        
        # Обратите внимание на {max_label_length} в фигурных скобках внутри f-строки
        print(f"{folder:<{max_label_length}} | {count:<6} | {bar}")
        count = i
    summ =sum(stats.values())
    print(f"\nВсего изображений: {summ}")
    print(f'Средне: {summ/count}')
# Укажите путь к вашей папке здесь
if __name__ == "__main__":
    path_to_check = "data/train"  # Замените на свой путь
    count_images_in_folders(path_to_check)