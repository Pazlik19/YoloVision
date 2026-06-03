"""
Отладка определения угла кубика/буквы и подбор HSV-диапазона оранжевого.

Запуск:
    python angle_debug.py path/to/cube.jpg

Слева — исходник с наложенным minAreaRect грани, главной осью буквы и углами.
Справа — текущая маска оранжевого. Трекбары сверху меняют HSV-порог в реальном времени.

Клавиши:  ESC — выход,  p — напечатать текущие HSV в консоль (для вставки в Coardinate.py).
"""
import argparse
import cv2
import numpy as np

from Coardinate import (
    get_angel,
    get_letter_angle,
    _orientation_360,
    ORANGE_LOWER,
    ORANGE_UPPER,
)

WIN = "angle debug  (ESC - выход, p - печать HSV)"
CTRL = "HSV controls"


def _nothing(_):
    pass


def _make_trackbars():
    cv2.namedWindow(CTRL, cv2.WINDOW_NORMAL)
    specs = [
        ("H min", ORANGE_LOWER[0], 179), ("H max", ORANGE_UPPER[0], 179),
        ("S min", ORANGE_LOWER[1], 255), ("S max", ORANGE_UPPER[1], 255),
        ("V min", ORANGE_LOWER[2], 255), ("V max", ORANGE_UPPER[2], 255),
    ]
    for name, val, mx in specs:
        cv2.createTrackbar(name, CTRL, int(val), mx, _nothing)


def _read_hsv():
    lo = np.array([cv2.getTrackbarPos("H min", CTRL),
                   cv2.getTrackbarPos("S min", CTRL),
                   cv2.getTrackbarPos("V min", CTRL)], dtype=np.uint8)
    hi = np.array([cv2.getTrackbarPos("H max", CTRL),
                   cv2.getTrackbarPos("S max", CTRL),
                   cv2.getTrackbarPos("V max", CTRL)], dtype=np.uint8)
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description="Отладка угла кубика/буквы и подбор HSV")
    ap.add_argument("image", help="путь к фото кубика (кроп или кадр)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ОШИБКА] Не удалось открыть изображение: {args.image}")
        return

    # Увеличим мелкие кропы / уменьшим крупные кадры до удобного размера
    h, w = img.shape[:2]
    scale = 400.0 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    _make_trackbars()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        lo, hi = _read_hsv()

        # Углы через те же функции, что использует сервер
        cube_angle = get_angel(img, lower=lo, upper=hi)
        letter_angle = get_letter_angle(img, lower=lo, upper=hi)

        # Маска оранжевого (как в get_angel)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        vis = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            face = max(contours, key=cv2.contourArea)
            box = cv2.boxPoints(cv2.minAreaRect(face)).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)

            # Стрелка направления буквы (по полному углу)
            cx, cy = face.reshape(-1, 2).mean(axis=0).astype(int)
            rad = np.radians(letter_angle)
            L = int(0.4 * max(vis.shape[:2]))
            tip = (int(cx + L * np.cos(rad)), int(cy + L * np.sin(rad)))
            cv2.arrowedLine(vis, (cx, cy), tip, (0, 180, 255), 2, tipLength=0.25)

        cv2.putText(vis, f"cube  (0-90):  {cube_angle}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(vis, f"letter(0-360): {letter_angle}", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

        combo = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(WIN, combo)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:                       # ESC
            break
        if key == ord('p'):
            print(f"ORANGE_LOWER = np.array({lo.tolist()}, dtype=np.uint8)")
            print(f"ORANGE_UPPER = np.array({hi.tolist()}, dtype=np.uint8)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
