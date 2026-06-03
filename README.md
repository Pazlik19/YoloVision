
# Система компьютерного зрения и навигации

Распознавание русских букв на кубиках в реальном времени с определением их
положения (координаты в мм) и угла поворота. Рассчитана на работу с роботом и
платой **Jetson Nano Dev Kit** (CSI-камера IMX219).

Конвейер: **фото → детекция объектов (YOLO) → классификация буквы (ResNet18)
→ перевод в реальные координаты + угол → телеметрия для робота.**

---

## Возможности

- Сбор датасета с камеры Jetson и отправка на ПК по SSH.
- Полуавтоматическая разметка (сортировка кропов по уверенности модели).
- Обучение классификатора **ResNet18** в две стадии (заморозка backbone → fine-tuning).
- Два режима инференса:
  - **yolo_resnet** — двухэтапный: YOLO детектит → ResNet классифицирует;
  - **yolo_only** — сквозной: одна YOLO детектит и классифицирует.
- Перевод пиксельных координат в миллиметры (по геометрии камеры) и расчёт угла поворота.
- Усреднение измерений объекта между кадрами + оценка погрешности (std).

---

## Структура проекта

| Папка / файл | Назначение |
|---|---|
| `steamPhoto.py` | Захват кадров с CSI-камеры (GStreamer), MJPEG-стрим (Flask), отправка снимков на ПК по SSH |
| `CNNModelTrain/` | Подготовка датасета и обучение классификатора (актуальная версия) |
| `CNN/` | Более ранняя версия скриптов обучения/сегментации/классификации |
| `main/` | Live-распознавание с веб-камеры на ПК (`YoloResNetVision4.py` — актуальная) |
| `JetsonNano/JetsonYolo3/` | Клиент-серверный инференс на устройстве (актуальная версия) |
| `Models/` | Веса моделей (`.pt`, `.pth`, `.engine`) — **не в git** |
| `YoloModelTrain/` | Датасет и запуски обучения YOLO |

> Модули `TextDetection2.py`, `Coardinate.py`, `bufer.py` намеренно скопированы в
> каждую папку развёртывания, чтобы каждый узел (ПК / Jetson) запускался автономно.

### Ключевые модули

- **`TextDetection2.py`** — класс `SymbolClassifier` (ResNet18): кроп → буква + уверенность.
- **`Coardinate.py`** — `get_real_coords()` (пиксели → мм через GSD камеры), `get_angel()` (угол грани кубика 0–90° по HSV-маске + `minAreaRect`) и `get_letter_angle()` (полный угол буквы 0–360° по её максимальному контуру).
- **`angle_debug.py`** — отладочный скрипт: подбор HSV-диапазона оранжевого и проверка углов на фото.
- **`bufer.py`** — `Scene` / `Letter`: накопление измерений по объекту между кадрами, усреднение, std.
- **`Segment.py`** — нарезка объектов с фото моделью YOLO в кропы.
- **`classifaer.py`** — раскладка кропов по классам; неуверенные → `low_confidence`.
- **`CalculeteImage.py`** — статистика баланса классов + выгрузка в Excel.

---

## Архитектура (поток данных)

```mermaid
flowchart TD
    subgraph CAPTURE["Сбор данных · Jetson Nano"]
        CAM[CSI-камера] --> STREAM[steamPhoto.py<br/>GStreamer + Flask MJPEG]
        STREAM -->|SSH / paramiko| RAW[(dataset_raw на ПК)]
    end

    subgraph DATASET["Подготовка датасета · ПК"]
        RAW --> SEG[Segment.py<br/>YOLO: вырезка объектов]
        SEG --> CROPS[(detected_objects)]
        CROPS --> SORT[classifaer.py<br/>сортировка по порогу]
        SORT --> LABELED[(data/train/latter_X)]
        LABELED --> STATS[CalculeteImage.py<br/>статистика + Excel]
    end

    subgraph TRAIN["Обучение · ПК + GPU"]
        LABELED --> PREP[prepare_datasets<br/>split + балансировка + аугментация]
        PREP --> RESNET[CNNPyTorchTrain_ResNet.py<br/>ResNet18, 2 стадии]
        RESNET --> WEIGHTS[(best_resnet_model.pth<br/>+ class_names.json)]
        RESNET -.export TensorRT.-> ENGINE[(best.engine)]
    end

    subgraph INFER["Распознавание в реальном времени"]
        WEIGHTS --> SERVER
        ENGINE --> SERVER[server.py · Docker<br/>ZMQ REP]
        JCLIENT[JetsonCameraWEB.py<br/>ZMQ REQ] -->|кадр JPEG| SERVER
        SERVER -->|кадр + JSON телеметрия| JCLIENT
        SERVER --> COORD[Coardinate.py<br/>пиксели → мм]
        SERVER --> BUF[bufer.py · Scene<br/>усреднение + погрешность]
        BUF --> ROBOT[Координаты для робота]
    end
```

## Двухэтапный режим распознавания (yolo_resnet)

```mermaid
sequenceDiagram
    participant C as Jetson (REQ)
    participant S as Server (REP)
    participant Y as YOLO detector
    participant R as ResNet (SymbolClassifier)

    C->>S: send(frame_bytes)
    S->>Y: detect(frame, conf>0.6)
    Y-->>S: boxes [x1,y1,x2,y2]
    loop по каждому объекту
        S->>S: crop = frame[y1:y2, x1:x2]
        S->>R: predict(crop)
        R-->>S: label, confidence
        alt confidence < 60%
            S->>S: "Low conf" (синяя рамка)
        else
            S->>S: label (зелёная рамка)
        end
        S->>S: get_real_coords → x,y (мм)
    end
    S-->>C: send_multipart([jpeg, json telemetry])
```

## Стадии обучения ResNet18

```mermaid
stateDiagram-v2
    [*] --> Подготовка
    Подготовка --> Стадия1: данные сбалансированы
    Стадия1: Стадия 1 (эпохи 1–5) · backbone заморожен, учится только fc, lr=1e-3
    Стадия1 --> Стадия2: epoch == freeze_epochs
    Стадия2: Стадия 2 (эпохи 6–30) · вся сеть, lr=1e-4, ReduceLROnPlateau
    Стадия2 --> Сохранение: val_acc > best_acc
    Сохранение --> Стадия2
    Стадия2 --> [*]: графики Loss/Acc
```

---

## Установка

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

На Jetson Nano `torch`/`torchvision`/`ultralytics` ставятся через JetPack или
docker-образ NVIDIA — см. `JetsonNano/Stup.txt`.

## Запуск

**Обучение классификатора (ПК):**
```bash
python CNNModelTrain/CNNPyTorchTrain_ResNet.py
```

**Статистика датасета:**
```bash
python CNNModelTrain/CalculeteImage.py
```

**Live-распознавание с веб-камеры (ПК):**
```bash
python main/YoloResNetVision4.py
```

**Инференс на Jetson Nano (в Docker):**
```bash
python3 JetsonYolo3/server.py --mode yolo_resnet   # или yolo_only
python3 JetsonYolo3/JetsonCameraWEB.py
```

---

## Параметры камеры (Jetson, IMX219)

Заданы в `Coardinate.py` / `config.py`:
- разрешение потока `1280×720`, фокус `f = 3.04 мм`;
- высота камеры над поверхностью `CAMERA_HEIGHT_MM` (в `config.py`);
- `CALIBRATION_FACTOR` — поправочный коэффициент по результатам калибровки.
