# Диаграммы системы компьютерного зрения и навигации

Графические материалы к пояснительной записке. Все схемы выполнены на языке
разметки **Mermaid** и отображаются на GitHub, в VS Code (расширение *Markdown
Preview Mermaid*) и в редакторе <https://mermaid.live> (для экспорта в PNG/SVG).

---

## Рисунок 1 — Общая архитектура системы (сквозной поток данных)

Полный жизненный цикл: сбор данных → обучение двух нейросетей (детектора
кубиков и классификатора букв) → инференс в реальном времени с построением
карты поля.

```mermaid
flowchart TD
    %% ================= СБОР ДАННЫХ =================
    subgraph CAP["Сбор данных · Jetson Nano"]
        CAM[CSI-камера IMX219] --> SP[steamPhoto.py<br/>GStreamer + Flask]
        SP -->|SSH / paramiko| RAW[(Фото поля с кубиками)]
    end

    %% ============ ОБУЧЕНИЕ ДЕТЕКТОРА YOLO ============
    subgraph YTR["Обучение детектора кубиков · ПК"]
        RAW --> SAM[autoDataSetDetection.py<br/>SAM 3: авторазметка 'the cube']
        SAM --> YDS[(yolo_dataset_sam3<br/>images + labels + data.yaml)]
        YDS --> SPL[SplitDataset.py<br/>train/val 80/20]
        SPL --> YT[YoloTrainDetect.py<br/>YOLO11n · 50 эпох]
        YT --> BEST[(best.pt)]
        BEST -.export.-> BENG[(best.engine · TensorRT)]
    end

    %% ========== ОБУЧЕНИЕ КЛАССИФИКАТОРА ResNet ==========
    subgraph CTR["Обучение классификатора букв · ПК"]
        RAW --> SEG[Segment.py<br/>нарезка кубиков детектором]
        BEST --> SEG
        SEG --> CR[(detected_objects)]
        CR --> CLS[classifaer.py<br/>сортировка по порогу]
        CLS --> DS[(data/train/latter_X)]
        DS --> CALC[CalculeteImage.py<br/>баланс классов → Excel]
        DS --> BAL[prepare_datasets<br/>баланс + аугментация]
        BAL --> RES[CNNPyTorchTrain_ResNet.py<br/>ResNet18 · 2 стадии]
        RES --> PTH[(best_resnet_model.pth<br/>class_names.json)]
    end

    %% ================= ИНФЕРЕНС =================
    subgraph INF["Инференс в реальном времени · Jetson Nano"]
        JC[JetsonCameraWEB.py<br/>ZMQ REQ + Flask] <-->|кадр / телеметрия| SRV[server.py · Docker<br/>ZMQ REP]
        BENG --> SRV
        PTH --> SRV
        SRV --> SCN[bufer.py · Scene<br/>карта поля + усреднение]
        SCN --> OUT[/data → JSON<br/>label, x, y, angle/]
        OUT --> ROBOT[Манипулятор / робот]
    end

    CALC -.качество датасета.-> BAL
```

---

## Рисунок 2 — Пайплайн обучения детектора кубиков (SAM 3 → YOLO)

Автоматическая разметка датасета без ручного выделения объектов: модель
сегментации **SAM 3** по текстовому запросу «the cube» строит маски кубиков,
которые пересчитываются в прямоугольники формата YOLO. Затем датасет
разбивается на выборки и используется для обучения детектора YOLO11n.

```mermaid
flowchart TD
    A[(Data/ — исходные фото поля)] --> B[autoDataSetDetection.py]

    subgraph SAMSTEP["Авторазметка через SAM 3"]
        B --> C[SAM3SemanticPredictor<br/>текстовый промпт 'the cube']
        C --> D[Маски сегментации кубиков]
        D --> E[Маска → bounding box<br/>формат YOLO: 0 xc yc w h]
        D --> F[Контрольная визуализация<br/>рамки на фото]
    end

    E --> G[(yolo_dataset_sam3<br/>images/ + labels/ + data.yaml)]
    F --> H[(visualized/ — ручная проверка)]
    G --> I[SplitDataset.py<br/>train/val = 80/20, seed=42]
    I --> J[(train, val<br/>обновлённый data.yaml)]
    J --> K[YoloTrainDetect.py<br/>YOLO11n · imgsz=640 · 50 эпох]
    K --> L[(runs/.../best.pt)]
    L -.export TensorRT.-> M[(best.engine)]
    M --> N[Детекция кубиков в server.py]
    L --> O[Нарезка датасета в Segment.py]
```

---

## Рисунок 3 — Двухэтапный режим распознавания (yolo_resnet)

Последовательность обработки одного кадра: детекция кубиков нейросетью YOLO,
покадровая классификация каждого кубика сетью ResNet18, фильтрация по
уверенности и формирование телеметрии.

```mermaid
sequenceDiagram
    participant C as Клиент (Jetson, REQ)
    participant S as Сервер (Docker, REP)
    participant Y as YOLO detector
    participant R as ResNet (SymbolClassifier)

    C->>S: send(кадр JPEG)
    S->>Y: detect(frame, conf > 0.6)
    Y-->>S: рамки [x1,y1,x2,y2]
    loop по каждому кубику
        S->>S: crop = frame[y1:y2, x1:x2]
        S->>R: predict(crop)
        R-->>S: буква, уверенность
        alt уверенность < 60%
            S->>S: "Low conf" (синяя рамка)
        else
            S->>S: буква + get_letter_angle + get_real_coords
        end
    end
    S-->>C: send_multipart([кадр, JSON-телеметрия])
    C->>C: Scene.update_scene(детекции, позиция камеры)
```

---

## Рисунок 4 — Стадии обучения классификатора ResNet18

Стратегия transfer learning: на первом этапе обучается только классификатор
(голова сети) при замороженном backbone, на втором — дообучается вся сеть с
пониженным шагом обучения и адаптивным планировщиком.

```mermaid
stateDiagram-v2
    [*] --> Подготовка
    Подготовка --> Стадия1: данные сбалансированы
    Стадия1: Стадия 1 (эпохи 1-5) · backbone заморожен, учится только fc, lr=1e-3
    Стадия1 --> Стадия2: epoch == freeze_epochs
    Стадия2: Стадия 2 (эпохи 6-30) · вся сеть, lr=1e-4, ReduceLROnPlateau
    Стадия2 --> Сохранение: val_acc > best_acc
    Сохранение --> Стадия2
    Стадия2 --> [*]: графики Loss / Accuracy
```

---

## Рисунок 5 — Диаграмма классов буфера сцены (bufer.py)

Структура накопления и усреднения измерений. Класс `Scene` хранит карту
обнаруженных объектов, выполняет ассоциацию новых детекций с уже известными
объектами и управляет их жизненным циклом. Класс `Letter` накапливает историю
измерений одного кубика и хранит усреднённые значения с оценкой погрешности.

```mermaid
classDiagram
    class Scene {
        +dict objects
        +int next_id
        +float distance_threshold
        +int max_misses
        +int min_hits
        +float fov_width
        +float fov_height
        +update_scene(detections, cam_x, cam_y)
        +get_all_objects_data() list
        +get_word() str
        -_get_distance(x1, y1, x2, y2) float
        -_is_in_fov(obj_x, obj_y, cam_x, cam_y) bool
    }

    class Letter {
        +int id
        +list~str~ labels
        +list~float~ x_vals
        +list~float~ y_vals
        +list~float~ angles
        +str label
        +float x_mid
        +float y_mid
        +float angle_mid
        +int hit_count
        +int miss_count
        +count() int
        +update(label, x, y, angle)
    }

    Scene "1" o-- "0..*" Letter : хранит
```

---

## Рисунок 6 — Жизненный цикл объекта в буфере сцены

Логика устойчивого отслеживания кубика на карте при перемещении камеры.
Объект подтверждается после нескольких совпадений, сохраняется при выходе из
поля зрения камеры и удаляется только как ложный — если должен быть в кадре,
но не детектируется заданное число раз подряд.

```mermaid
stateDiagram-v2
    [*] --> Новый: первая детекция
    Новый --> Кандидат: создан Letter (next_id)
    Кандидат --> Подтверждён: hit_count >= min_hits
    Подтверждён --> Подтверждён: совпадение (update, miss_count=0)

    Подтверждён --> ВнеКадра: камера уехала (не в FOV)
    ВнеКадра --> Подтверждён: камера вернулась, снова детектится
    note right of ВнеКадра : Хранится в глобальной карте, не удаляется

    Подтверждён --> Пропущен: в FOV, но не детектится
    Пропущен --> Подтверждён: снова найден (miss_count=0)
    Пропущен --> [*]: miss_count >= max_misses (удалён как ложный)
```

---

## Рисунок 7 — Зависимости модулей узла инференса (JetsonYolo3)

Граф импортов программных модулей серверной и клиентской частей. Внешние
библиотеки выделены отдельно. Демонстрирует разделение ответственности:
детекция и классификация — на сервере, накопление карты — на клиенте.

```mermaid
flowchart LR
    subgraph EXT["Внешние библиотеки"]
        YOLOlib[ultralytics YOLO]
        TORCH[torch / torchvision]
        CV[OpenCV]
        ZMQ[pyzmq]
        FLASK[Flask]
    end

    SRV[server.py] --> CFG[config.py]
    SRV --> DRAW[draw_utils.py]
    SRV --> COORD[Coardinate.py]
    SRV --> TD[TextDetection2.py]
    SRV --> YOLOlib
    SRV --> ZMQ

    COORD --> CV
    DRAW --> CV
    TD --> TORCH

    CLIENT[JetsonCameraWEB.py] --> BUF[bufer.py]
    CLIENT --> ZMQ
    CLIENT --> FLASK
    CLIENT --> CV

    SRV -. ZMQ REQ/REP .- CLIENT
```

---

## Рисунок 8 — Определение угла поворота кубика и буквы (Coardinate.py)

Алгоритм вычисления ориентации. Угол грани кубика определяется по силуэту
оранжевой грани (устойчиво, по модулю 90°), полный угол буквы — по её
максимальному контуру методом главных компонент.

```mermaid
flowchart TD
    IN[crop кубика BGR] --> HSV[Перевод в HSV<br/>маска оранжевого]
    HSV --> MORPH[Морфология:<br/>close + open]
    MORPH --> CNT{Контур грани<br/>>= 5% площади?}
    CNT -- нет --> Z[угол = 0]
    CNT -- да --> RECT[cv2.minAreaRect<br/>угол грани 0-90°]

    MORPH --> DARK[Тёмные пиксели внутри грани<br/>порог = mean − 0.7·σ]
    DARK --> LCNT{Контур буквы<br/>найден?}
    LCNT -- нет --> RECT
    LCNT -- да --> MAX[Максимальный контур]
    MAX --> PCA[PCA: главная ось]
    PCA --> SKEW[Направление по асимметрии<br/>3-й момент → 0-360°]
    SKEW --> OUT[Угол буквы 0-360°]
```

---

## Рисунок 9 — Диаграмма вариантов использования (use-case)

Функции системы с точки зрения действующих лиц: оператора, обслуживающего
работу установки, инженера-разработчика, готовящего модели, и робота-
манипулятора как потребителя телеметрии.

```mermaid
flowchart LR
    OP([Оператор])
    DEV([Инженер-разработчик])
    ROB([Робот-манипулятор])

    subgraph SYS["Система компьютерного зрения"]
        U1(["Запустить распознавание"])
        U2(["Просмотреть видеопоток"])
        U3(["Получить карту объектов JSON"])
        U4(["Собрать датасет с камеры"])
        U5(["Обучить детектор кубиков<br/>SAM 3 + YOLO"])
        U6(["Обучить классификатор букв"])
        U7(["Откалибровать камеру и пороги"])
        U8(["Получить координаты и угол кубика"])
    end

    OP --- U1
    OP --- U2
    OP --- U3
    DEV --- U4
    DEV --- U5
    DEV --- U6
    DEV --- U7
    ROB --- U8
    U3 -.поставляет данные.-> U8
```

---

## Рисунок 10 — Диаграмма развёртывания

Физическое размещение компонентов по узлам и протоколы взаимодействия между
ними. Тяжёлый инференс изолирован в Docker-контейнере с поддержкой TensorRT,
захват кадров и накопление карты выполняются на хост-системе Jetson Nano.

```mermaid
flowchart TB
    subgraph PC["💻 ПК (обучение)"]
        SAMNODE[autoDataSetDetection.py<br/>SAM 3 авторазметка]
        YOLOTRAIN[YoloTrainDetect.py<br/>обучение YOLO11n]
        RESTRAIN[CNNPyTorchTrain_ResNet.py<br/>обучение ResNet18]
        WEIGHTS[(Веса: best.engine + .pth)]
    end

    subgraph JETSON["🟩 Jetson Nano (хост)"]
        CAMNODE[CSI-камера IMX219]
        CLIENTNODE[JetsonCameraWEB.py<br/>захват + Scene + Flask]
        subgraph DOCKER["🐳 Docker-контейнер"]
            SERVERNODE[server.py<br/>YOLO + ResNet + TensorRT]
        end
    end

    BROWSER[🌐 Браузер оператора]
    ROBOTNODE[🤖 Контроллер манипулятора]

    SAMNODE --> YOLOTRAIN
    CAMNODE -->|GStreamer / V4L2| CLIENTNODE
    CLIENTNODE <-->|ZMQ REQ/REP :5555| SERVERNODE
    PC -.SCP / SSH: копирование весов.-> DOCKER
    WEIGHTS -.-> SERVERNODE
    CLIENTNODE -->|HTTP MJPEG + /data JSON :5000| BROWSER
    CLIENTNODE -->|телеметрия x, y, angle| ROBOTNODE
```

---

## Рисунок 11 — Блок-схема алгоритма главного цикла сервера (ГОСТ 19.701-90)

Алгоритм обработки видеопотока серверной частью: инициализация моделей,
приём кадра, детекция, покадровая классификация и измерение каждого объекта,
формирование и отправка телеметрии.

```mermaid
flowchart TD
    A([Начало]) --> B[/Аргумент: режим работы/]
    B --> C[Загрузка моделей<br/>YOLO + ResNet/TensorRT]
    C --> D[Привязка ZMQ-сокета :5555]
    D --> E[/Приём кадра от клиента/]
    E --> F{Кадр<br/>декодирован?}
    F -- Нет --> G[/Отправить ERROR/]
    G --> E
    F -- Да --> H[Детекция объектов YOLO]
    H --> I{Есть<br/>необработанные<br/>объекты?}
    I -- Да --> J[Вырезать crop объекта]
    J --> K[Классификация буквы ResNet]
    K --> L[get_letter_angle:<br/>угол поворота]
    L --> M[get_real_coords:<br/>координаты в мм]
    M --> N[Добавить объект в телеметрию<br/>+ отрисовать рамку]
    N --> I
    I -- Нет --> O[Кодировать кадр в JPEG]
    O --> P[/Отправить кадр + JSON/]
    P --> E
```
