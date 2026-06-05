import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import json
import numpy as np
import cv2
import os
import time
from pathlib import Path
MODEL_DIR = Path(__file__).parent.parent / "Models"

# Верхняя граница батча TRT-движка. ДОЛЖНА совпадать с max_batch_size в
# convertResnetTensor.py — движок собран с этим лимитом, при превышении он упадёт.
TRT_MAX_BATCH = 32


class SymbolClassifier:
    def __init__(self, model_path= MODEL_DIR/'best_resnet_model.pth',  config_path= MODEL_DIR/'class_names.json',
                 trt_path= MODEL_DIR/'best_resnet_trt.pth'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфиг не найден: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Приоритет — TensorRT-движок (best_resnet_trt.pth, собирается на Jetson через
        # convertResnetTensor.py). Если его нет или torch2trt/CUDA недоступны (например,
        # на ПК для разработки) — откатываемся на обычный ResNet18 (.pth).
        self.use_trt = False
        if os.path.exists(trt_path) and self.device.type == "cuda":
            try:
                from torch2trt import TRTModule
                self.model = TRTModule()
                self.model.load_state_dict(torch.load(trt_path))
                self.use_trt = True
                print(f"[SymbolClassifier] TensorRT-движок загружен: {os.path.basename(str(trt_path))}")
            except Exception as e:
                print(f"[SymbolClassifier] Не удалось загрузить TRT ({e}). Откат на ResNet18 .pth")
                self.use_trt = False

        if not self.use_trt:
            # Инициализируем ResNet18 (как при обучении)
            self.model = models.resnet18()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, len(self.class_names))

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Веса не найдены: {model_path}")

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device).eval()
            print(f"[SymbolClassifier] Загружена PyTorch-модель ResNet18 (без TensorRT)")

        # Трансформации строго как при обучении ResNet
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Тайминги последнего вызова predict_batch (для инструментации в server.py)
        self.last_timings = {"preprocess_ms": 0.0, "forward_ms": 0.0}

    def predict(self, input_img):
        if isinstance(input_img, str):
            img = Image.open(input_img)
        elif isinstance(input_img, np.ndarray):
            if len(input_img.shape) == 3:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(input_img)
        else:
            img = input_img
            
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return self.class_names[pred.item()], conf.item() * 100

    def predict_batch(self, crops_bgr):
        """
        Пакетная классификация списка кропов одним forward-проходом.

        crops_bgr : список BGR-кадров (как frame[y1:y2, x1:x2] из OpenCV).
        return    : список кортежей (label, confidence%) в исходном порядке.

        Зачем батч: на Jetson Nano оверхед запуска CUDA-ядра и Python-цикла велик
        относительно самой работы ResNet18. Прогон N кропов одним тензором
        [N, 3, 224, 224] убирает линейный рост времени по числу объектов.
        Конвертация BGR->RGB делается здесь РОВНО один раз (без двойного свопа,
        который был при вызове predict() с уже сконвертированным кропом).
        """
        if not crops_bgr:
            self.last_timings = {"preprocess_ms": 0.0, "forward_ms": 0.0}
            return []

        # --- Стадия 1: препроцессинг (CPU, PIL, по одному кропу) ---
        t0 = time.time()
        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.preprocess(Image.fromarray(rgb)))
        batch = torch.stack(tensors).to(self.device)
        t1 = time.time()

        # --- Стадия 2: forward (GPU/TRT) ---
        # TRT-движок собран с фиксированным max_batch_size — если объектов больше,
        # режем на чанки. Для обычной .pth-модели лимита нет (один проход).
        chunk = TRT_MAX_BATCH if self.use_trt else len(batch)

        confs_all, preds_all = [], []
        with torch.no_grad():
            for start in range(0, len(batch), chunk):
                outputs = self.model(batch[start:start + chunk])
                probs = torch.nn.functional.softmax(outputs, dim=1)
                c, p = torch.max(probs, dim=1)
                confs_all.append(c)
                preds_all.append(p)

        # GPU-вызовы асинхронные — синхронизируемся, чтобы forward_ms был реальным.
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()

        self.last_timings = {
            "preprocess_ms": (t1 - t0) * 1000,
            "forward_ms": (t2 - t1) * 1000,
        }

        confs = torch.cat(confs_all)
        preds = torch.cat(preds_all)
        return [(self.class_names[p.item()], c.item() * 100)
                for p, c in zip(preds, confs)]