import os
from pathlib import Path
import pandas as pd

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
            img_count = sum(1 for f in item.iterdir() if f.suffix.lower() in valid_extensions)
            stats[item.name] = img_count

    if not stats:
        print("Изображения или подпапки не найдены.")
        return

    # Вычисляем общую сумму и среднее значение
    total_images = sum(stats.values())
    total_folders = len(stats)
    avg_images = total_images / total_folders

    # Подготовка данных для консоли и Excel
    max_label_length = max(len(name) for name in stats.keys())
    if max_label_length < 5:
        max_label_length = 5
        
    max_count = max(stats.values()) if stats.values() else 1
    
    # Шапка таблицы в консоли
    print(f"{'Папка':<{max_label_length}} | {'Кол-во':<6} | {'Откл. от ср.':<12} | График")
    print("-" * (max_label_length + 35))
    
    # Списки для сборки датафрейма Excel
    excel_data = []
    
    for folder, img_count in sorted(stats.items()):
        # Считаем точное отклонение
        deviation = img_count - avg_images
        
        # Строковое представление для консоли (с плюсом или минусом)
        if deviation > 0:
            dev_str = f"-{abs(deviation):.1f}"  # Нужно убавить
            excel_dev = -round(abs(deviation), 1)
        elif deviation < 0:
            dev_str = f"+{abs(deviation):.1f}"  # Нужно добавить
            excel_dev = round(abs(deviation), 1)
        else:
            dev_str = "0.0"
            excel_dev = 0.0
            
        # Рисуем полоску графика для консоли
        bar_length = int((img_count / max_count) * 30) if max_count > 0 else 0
        bar = "█" * bar_length
        
        print(f"{folder:<{max_label_length}} | {img_count:<6} | {dev_str:<12} | {bar}")
        
        # Добавляем чистые данные в список для Excel
        excel_data.append({
            "Название папки": folder,
            "Количество изображений": img_count,
            "Действие (Отклонение)": excel_dev
        })
        
    print("-" * (max_label_length + 35))
    print(f"Всего папок: {total_folders}")
    print(f"Всего изображений: {total_images}")
    print(f"Среднее количество: {avg_images:.2f}")

    # --- СЕКЦИЯ ГЕНЕРАЦИИ EXCEL ---
    # Создаем DataFrame из собранных данных
    df = pd.DataFrame(excel_data)
    
    # Добавляем строку "Итого" и "Среднее" в самый конец Excel-таблицы
    summary_rows = pd.DataFrame([
        {"Название папки": "ВСЕГО ИЗОБРАЖЕНИЙ:", "Количество изображений": total_images, "Действие (Отклонение)": ""},
        {"Название папки": "СРЕДНЕЕ НА ПАПКУ:", "Количество изображений": round(avg_images, 2), "Действие (Отклонение)": ""}
    ])
    df = pd.concat([df, summary_rows], ignore_index=True)
    
    # Путь для сохранения Excel (в папку со скриптом)
    excel_path = Path(__file__).parent / "dataset_stats.xlsx"
    
    # Сохраняем в файл
    df.to_excel(excel_path, index=False, sheet_name="Статистика датасета")
    print(f"\n[INFO] Excel таблица успешно сохранена: {excel_path.name}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    path_to_check = script_dir / "data/train"  
    
    count_images_in_folders(path_to_check)