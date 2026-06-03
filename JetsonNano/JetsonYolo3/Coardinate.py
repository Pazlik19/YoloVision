import cv2
import numpy as np
import os
def get_real_coords(u, v, H):
    """
    Перевод пиксельных координат (u, v) в метрические (x_real, y_real) на сцене.
    H - высота камеры над поверхностью (в мм), берется из config.CAMERA_HEIGHT_MM (200 мм)
    """
    # Параметры сенсора Sony IMX219 для режима 1280x720 (с учетом кропа под 16:9)
    W_p, H_p = 1280, 720       # Разрешение трансляции в пикселях
    W_s = 3.674                # Физическая ширина используемой матрицы (мм)
    H_s = 2.066                # Физическая высота используемой матрицы (мм) с учетом кропа 16:9
    f = 3.04                   # Реальное фокусное расстояние объектива IMX219 (мм)

    # 1. Находим оптический центр изображения (привязано к разрешению)
    u_center = W_p / 2.0
    v_center = H_p / 2.0

    # 2. Вычисляем GSD (Размер одного пикселя на объекте в мм)
    # Формула: (H * размер_сенсора) / (f * разрешение_в_пикселях)
    gsd_x = (H * W_s) / (f * W_p)
    gsd_y = (H * H_s) / (f * H_p)

    # 3. Вычисляем смещение пикселя относительно оптического центра
    delta_u = u - u_center
    delta_v = v_center - v  # Инвертируем ось V, чтобы Y рос "вверх" по кадру от робота

    # 4. Переводим пиксели в базовые реальные миллиметры на плоскости
    x_real_raw = delta_u * gsd_x
    y_real_raw = delta_v * gsd_y

    # 5. КОРРЕКЦИЯ КАЛИБРОВКИ (Оптический коэффициент линзы)
    # Реальное расстояние (30 мм) / Измеренное расстояние (18 мм) = 1.6666...
    CALIBRATION_FACTOR = 1.667
    
    x_real = x_real_raw * CALIBRATION_FACTOR
    y_real = y_real_raw * CALIBRATION_FACTOR

    # Возвращаем округленные до 2 знаков координаты
    return round(x_real, 2), round(y_real, 2)
# Тебе нужно будет заранее сохранить идеальные, ровные картинки каждой буквы
# в папку "templates" (например, A.jpg, B.jpg и т.д.)
TEMPLATES_DIR = "templates/"

# Словарь для кэширования шаблонов, чтобы не читать их с диска каждый кадр
loaded_templates = {}
def get_angel(crop, label, templates_dir="templates"): # 1. Исправили регистр папки по умолчанию
    """
    Повышенная точность расчета угла поворота объекта (до 1 градуса).
    Приводит изображения к квадрату во избежание обрезки краев.
    """
    if len(crop.shape) == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop.copy()
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    crop_gray = clahe.apply(crop_gray)

    # 2. Загрузка шаблона
    template_path = f"{templates_dir}/{label}.jpg" 
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None:
        print(f"Предупреждение: Шаблон для класса '{label}' не найден по пути: {template_path}")
        return 0

    # Сделай фиксированный размер-квадрат, чтобы при вращении углы букв не срезались
    SQUARE_SIZE = 128
    crop_gray = cv2.resize(crop_gray, (SQUARE_SIZE, SQUARE_SIZE))
    template_resized = cv2.resize(template, (SQUARE_SIZE, SQUARE_SIZE))
    template_resized = clahe.apply(template_resized)

    # Вспомогательная функция для поворота квадратного изображения
    def rotate_image(image, angle):
        center = (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, (SQUARE_SIZE, SQUARE_SIZE), flags=cv2.INTER_LINEAR)

    # --- ЭТАП 1: ГРУБЫЙ ПОИСК (0, 90, 180, 270) ---
    best_coarse_angle = 0
    max_coarse_val = -1
    
    for angle in [0, 90, 180, 270]:
        rotated_template = rotate_image(template_resized, angle)
        res = cv2.matchTemplate(crop_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > max_coarse_val:
            max_coarse_val = max_val
            best_coarse_angle = angle

    # --- ЭТАП 2: ТОЧНЫЙ ПОИСК (±45° с шагом 1°) ---
    best_fine_angle = best_coarse_angle
    max_fine_val = max_coarse_val
    
    start_angle = best_coarse_angle - 45
    end_angle = best_coarse_angle + 45
    
    for angle in range(start_angle, end_angle + 1):
        if angle in [0, 90, 180, 270] and angle != best_coarse_angle:
            continue
            
        rotated_template = rotate_image(template_resized, angle)
        res = cv2.matchTemplate(crop_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > max_fine_val:
            max_fine_val = max_val
            best_fine_angle = angle

    return best_fine_angle % 360
# Пример использования:
# Если объект в центре кадра (1640, 1232) на высоте 1000 мм:

print(get_real_coords(1640, 1232, 1000)) # Результат: (0.0, 0.0)

# Если объект смещен к краю (например, точка 2000, 500) на высоте 1000 мм:
x, y = get_real_coords(2000, 500, 1000)
print(f"Координаты объекта: X = {x} мм, Y = {y} мм")