"""
Конвертация классификатора ResNet18 (.pth) -> TensorRT-движок через torch2trt.

ВАЖНО: запускать ТОЛЬКО на самом Jetson Nano (внутри Docker-контейнера
ultralytics:latest-jetson-jetpack4). TensorRT-движок привязан к архитектуре GPU
и версии TensorRT — собранный на ПК (RTX/CUDA 12.x) на Jetson НЕ запустится.

Что делает скрипт:
  1. Поднимает ResNet18 ровно как при обучении/инференсе (fc на N классов из
     class_names.json) и грузит веса best_resnet_model.pth.
  2. Конвертирует forward модели в TensorRT (FP16) через torch2trt.
  3. Прогоняет проверку: сравнивает выходы исходной и TRT-модели (макс. отличие
     логитов + совпадение предсказанных классов) на батче.
  4. Сохраняет state_dict TRT-модели в best_resnet_trt.pth.

На сервере (TextDetection2.py) движок потом грузится так:
    from torch2trt import TRTModule
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load("best_resnet_trt.pth"))
и зовётся как обычная модель: model_trt(batch_tensor).

Запуск на Jetson:
    python3 convertResnetTensor.py
"""

from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
from torchvision import models

# torch2trt ставится отдельно (в jetpack-образе обычно уже есть). Если нет:
#   git clone https://github.com/NVIDIA-AI-IOT/torch2trt
#   cd torch2trt && python3 setup.py install
try:
    from torch2trt import torch2trt, TRTModule
except ImportError:
    print("[!] Не найден torch2trt. Установи его НА JETSON:")
    print("    git clone https://github.com/NVIDIA-AI-IOT/torch2trt")
    print("    cd torch2trt && python3 setup.py install")
    sys.exit(1)


# --- НАСТРОЙКИ (при необходимости поправь пути) ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best_resnet_model.pth"   # исходные веса ResNet18
CLASS_NAMES_PATH = SCRIPT_DIR / "class_names.json"  # список классов (определяет размер fc)
OUTPUT_PATH = SCRIPT_DIR / "best_resnet_trt.pth"    # куда сохранить TRT-движок

INPUT_SIZE = 224        # размер входа (как при обучении)
MAX_BATCH_SIZE = 32     # максимальное число объектов в кадре за один forward
USE_FP16 = True         # FP16 — как у YOLO (half=True). Maxwell-GPU Nano даёт ~2x


def build_model(num_classes: int) -> nn.Module:
    """ResNet18 с fc на num_classes — идентично TextDetection2.SymbolClassifier."""
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    if not torch.cuda.is_available():
        print("[!] CUDA недоступна. Скрипт должен выполняться на Jetson с рабочим GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"--- Конвертация ResNet18 -> TensorRT ---")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Классы (размер выходного слоя)
    if not CLASS_NAMES_PATH.exists():
        print(f"[!] Не найден {CLASS_NAMES_PATH}")
        sys.exit(1)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Классов: {num_classes}")

    # 2. Модель + веса
    if not MODEL_PATH.exists():
        print(f"[!] Не найден {MODEL_PATH}")
        sys.exit(1)
    model = build_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print(f"Веса загружены: {MODEL_PATH.name}")

    # 3. Конвертация в TensorRT.
    #    example input — фиктивный тензор нужной формы (значения не важны для FP16,
    #    калибровка как при INT8 не требуется). max_batch_size задаёт верхнюю
    #    границу батча: в кадре объектов всегда меньше, но ставим с запасом.
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    print(f"Сборка движка (fp16={USE_FP16}, max_batch={MAX_BATCH_SIZE})... это займёт пару минут.")
    model_trt = torch2trt(
        model,
        [example],
        fp16_mode=USE_FP16,
        max_batch_size=MAX_BATCH_SIZE,
    )
    print("Движок собран.")

    # 4. Проверка: исходная модель vs TRT на батче (>1, чтобы проверить и батчинг).
    test_batch = torch.randn(min(4, MAX_BATCH_SIZE), 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        y_ref = model(test_batch)
        y_trt = model_trt(test_batch)

    max_abs_diff = (y_ref - y_trt).abs().max().item()
    same_pred = bool((y_ref.argmax(dim=1) == y_trt.argmax(dim=1)).all().item())
    print(f"Проверка: макс. отличие логитов = {max_abs_diff:.4f}, "
          f"предсказания совпадают = {same_pred}")
    if not same_pred:
        print("[!] Внимание: предсказания TRT и исходной модели расходятся. "
              "Для FP16 небольшое отличие логитов нормально, но разные классы — повод "
              "перепроверить (попробуй USE_FP16=False для диагностики).")

    # 5. Сохранение
    torch.save(model_trt.state_dict(), OUTPUT_PATH)
    print(f"Готово! TRT-движок сохранён: {OUTPUT_PATH}")

    # 6. Контрольная загрузка тем же способом, что и на сервере
    reloaded = TRTModule()
    reloaded.load_state_dict(torch.load(OUTPUT_PATH))
    with torch.no_grad():
        y_reloaded = reloaded(test_batch)
    reload_ok = bool((y_reloaded.argmax(dim=1) == y_trt.argmax(dim=1)).all().item())
    print(f"Контрольная перезагрузка движка: {'OK' if reload_ok else 'ОШИБКА'}")


if __name__ == "__main__":
    main()
