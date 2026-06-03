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
# --- Определение угла поворота оранжевого кубика ---
# Кубик квадратный, поэтому угол определяется по модулю 90° (для захвата
# манипулятором этого достаточно — схват одинаков в 4 ориентациях).
# Метод: выделяем кубик маской по цвету в HSV -> берём минимальный
# описывающий прямоугольник (cv2.minAreaRect) -> нормируем угол в [0, 90).
# Это в сотни раз быстрее перебора поворотов шаблона и не требует шаблонов.

# Диапазон оранжевого в HSV (OpenCV: H 0..179). Подстрой под своё освещение.
ORANGE_LOWER = np.array([5, 80, 80], dtype=np.uint8)
ORANGE_UPPER = np.array([25, 255, 255], dtype=np.uint8)


def _normalize_angle_90(angle: float) -> float:
    """Приводит угол к диапазону [0, 90) — учёт симметрии квадрата."""
    angle = angle % 90.0
    if angle < 0:
        angle += 90.0
    return angle


def get_angel(crop, label=None, lower=ORANGE_LOWER, upper=ORANGE_UPPER):
    """
    Угол поворота кубика в кадре по его оранжевой грани.

    crop  : BGR-изображение вырезанного кубика (как из frame[y1:y2, x1:x2]).
    label  : не используется (оставлен для совместимости со старым вызовом).
    return: угол в градусах [0, 90), или 0 если кубик не выделился.
    """
    if crop is None or crop.size == 0 or len(crop.shape) != 3:
        return 0

    # 1. Маска оранжевого в HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # 2. Чистим шум (закрываем тени от выдавленной буквы внутри грани)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Крупнейший контур = грань кубика
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    largest = max(contours, key=cv2.contourArea)
    # Отсекаем мусор: контур должен занимать заметную часть кропа
    if cv2.contourArea(largest) < 0.05 * crop.shape[0] * crop.shape[1]:
        return 0

    # 4. Угол минимального описывающего прямоугольника
    (_, _), (_, _), angle = cv2.minAreaRect(largest)

    return round(_normalize_angle_90(angle), 1)


def _orientation_360(contour):
    """
    Угол главной оси контура (PCA) с разрешением направления через 3-й момент.
    Возвращает [0, 360). Отсчёт против часовой от оси +X в координатах изображения.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return 0.0

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)          # по возрастанию собственных значений
    major = eigvecs[:, int(np.argmax(eigvals))]     # главная ось — максимальная дисперсия

    # Проекция точек на главную ось: знак асимметрии (skewness) задаёт направление,
    # чтобы различить букву и её поворот на 180°.
    proj = centered @ major
    if np.mean(proj ** 3) < 0:
        major = -major

    angle = np.degrees(np.arctan2(major[1], major[0]))
    return float(angle % 360.0)


def get_letter_angle(crop, lower=ORANGE_LOWER, upper=ORANGE_UPPER):
    """
    Полный угол поворота БУКВЫ [0, 360) по её максимальному контуру.

    Буква выдавлена на оранжевой грани и видна как более тёмная область (тень в канавке).
    Шаги: выделяем грань кубика -> внутри неё ищем тёмный контур буквы ->
    берём максимальный контур -> главная ось через PCA + направление по асимметрии.
    Если букву выделить не удалось — откатываемся на угол грани кубика get_angel (0..90).
    """
    if crop is None or crop.size == 0 or len(crop.shape) != 3:
        return 0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    cube_mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if cv2.countNonZero(cube_mask) < 0.05 * crop.shape[0] * crop.shape[1]:
        return get_angel(crop, lower=lower, upper=upper)

    # Внутри грани буква = тёмные пиксели. Порог = среднее − 0.7·σ по области грани.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cube_pixels = gray[cube_mask > 0]
    thr = int(np.clip(cube_pixels.mean() - 0.7 * cube_pixels.std(), 1, 254))

    letter_mask = cv2.inRange(gray, 0, thr)
    letter_mask = cv2.bitwise_and(letter_mask, cube_mask)
    letter_mask = cv2.morphologyEx(letter_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # ОТСЛЕЖИВАНИЕ МАКСИМАЛЬНОГО КОНТУРА буквы
    contours, _ = cv2.findContours(letter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return get_angel(crop, lower=lower, upper=upper)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.01 * crop.shape[0] * crop.shape[1]:
        return get_angel(crop, lower=lower, upper=upper)

    return round(_orientation_360(largest), 1)


if __name__ == "__main__":
    # Демонстрация перевода пиксельных координат в метрические.
    # Запускается только при прямом вызове файла, а не при импорте на сервере.
    print(get_real_coords(1640, 1232, 1000))
    x, y = get_real_coords(2000, 500, 1000)
    print(f"Координаты объекта: X = {x} мм, Y = {y} мм")