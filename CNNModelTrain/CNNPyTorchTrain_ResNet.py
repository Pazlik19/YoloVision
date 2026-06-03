import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import shutil
from pathlib import Path 
from PIL import Image
import random

# --- НАСТРОЙКА ПУТЕЙ ---
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "data/train"               # Исходный сырой датасет
TRAIN_BALANCED_DIR = BASE_DIR / "data/train_balanced" # Папка для сбалансированного обучения
VAL_CLEAN_DIR = BASE_DIR / "data/val_clean"           # Папка для чистой валидации

TARGET_DIR = BASE_DIR / "parent.parent/Models" if 'parent.parent' in str(Path(__file__).parent.parent) else BASE_DIR / "Models"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# --- ТРАНСФОРМАЦИИ ---
# Для генерации аугментированных картинок на диск (PIL -> PIL)
aug_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(90), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3)
])

# Для подачи картинок в нейросеть во время обучения (перевод в тензоры и нормализация)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def prepare_datasets(src_path, train_dest, val_dest, extra_margin=10, val_ratio=0.2):
    """
    Разделяет оригинальные картинки на Train/Val ДО аугментации.
    Балансирует и аугментирует ТОЛЬКО обучающую выборку.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Исходная папка {src_path} не найдена!")
        
    # Очищаем старые папки генерации, если они были
    if train_dest.exists(): shutil.rmtree(train_dest)
    if val_dest.exists(): shutil.rmtree(val_dest)
    train_dest.mkdir(parents=True, exist_ok=True)
    val_dest.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    class_counts = {}
    
    # Собираем пути ко всем оригиналам
    for folder in src_path.iterdir():
        if folder.is_dir():
            imgs = [f for f in folder.iterdir() if f.suffix.lower() in valid_extensions]
            if imgs:
                class_counts[folder.name] = imgs

    if not class_counts:
        print("[ОШИБКА] Подпапки с изображениями не найдены.")
        return

    train_imgs_per_class = {}
    val_imgs_per_class = {}

    # Шаг 1: Разделяем оригиналы на train и val раздельно для каждого класса
    for class_name, img_paths in class_counts.items():
        random.seed(42) # Фиксируем сид для воспроизводимости сплита
        random.shuffle(img_paths)
        
        val_size = int(len(img_paths) * val_ratio)
        if val_size == 0 and len(img_paths) > 1:
            val_size = 1 # Гарантируем хотя бы 1 картинку в валидацию
            
        val_imgs_per_class[class_name] = img_paths[:val_size]
        train_imgs_per_class[class_name] = img_paths[val_size:]

    # Шаг 2: Находим максимальный размер класса ИМЕННО в обучении
    max_train_class_size = max(len(imgs) for imgs in train_imgs_per_class.values())
    target_train_size = max_train_class_size + extra_margin
    
    print(f"--- Подготовка и Разделение Данных ---")
    print(f"Макс. оригиналов в одном классе для обучения: {max_train_class_size}")
    print(f"Целевой размер классов для обучения (Max + {extra_margin}): {target_train_size}\n")

    # Шаг 3: Копируем чистую валидацию (БЕЗ аугментаций)
    for class_name, img_paths in val_imgs_per_class.items():
        class_val_dir = val_dest / class_name
        class_val_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_paths:
            shutil.copy(img_path, class_val_dir / img_path.name)

    # Шаг 4: Копируем оригиналы обучения + догенерируем аугментированные копии
    for class_name, img_paths in train_imgs_per_class.items():
        class_train_dir = train_dest / class_name
        class_train_dir.mkdir(parents=True, exist_ok=True)
        
        # Копируем оригиналы обучения
        for img_path in img_paths:
            shutil.copy(img_path, class_train_dir / img_path.name)
            
        current_count = len(img_paths)
        needed_count = target_train_size - current_count
        
        print(f"Класс '{class_name}': Обучение = {current_count} ориг. + {needed_count} аугм. | Валидация = {len(val_imgs_per_class[class_name])} ориг.")
        
        # Генерируем недостающие картинки
        for i in range(needed_count):
            random_img_path = random.choice(img_paths)
            with Image.open(random_img_path) as img:
                augmented_img = aug_transform(img)
                gen_name = f"aug_{i}_{random_img_path.stem}.jpg"
                augmented_img.save(class_train_dir / gen_name, "JPEG")
                
    print(f"\n[УСПЕХ] Данные успешно разделены и сбалансированы!")


# Запуск разделения и генерации
prepare_datasets(SRC_DIR, TRAIN_BALANCED_DIR, VAL_CLEAN_DIR, extra_margin=10, val_ratio=0.2)

# --- ИНИЦИАЛИЗАЦИЯ ЗАГРУЗЧИКОВ ДАННЫХ ---
train_dataset = datasets.ImageFolder(root=str(TRAIN_BALANCED_DIR), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(VAL_CLEAN_DIR), transform=train_transform)

# Сохраняем имена классов
with open(TARGET_DIR / 'class_names.json', 'w', encoding='utf-8') as f:
    json.dump(train_dataset.classes, f)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --- АРХИТЕКТУРА МОДЕЛИ ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))


# --- ФУНКЦИЯ ДВУХЭТАПНОГО ОБУЧЕНИЯ ---
def train_two_stages(model, train_loader, val_loader, total_epochs=30, freeze_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\nИспользуется устройство: {device}")
    
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    # === ЭТАП 1: Замораживаем бэкбон, обучаем только голову ===
    print(f"\n=== СТАДИЯ 1: Обучение только классификатора ({freeze_epochs} эпох) ===")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = None
    is_fine_tuning = False

    for epoch in range(total_epochs):
        # Переключение на СТАДИЮ 2 (Тонкая настройка всей сети)
        if epoch == freeze_epochs:
            print(f"\n=== СТАДИЯ 2: Тонкая настройка всей сети ({total_epochs - freeze_epochs} эпох) ===")
            for param in model.parameters():
                param.requires_grad = True
            
            # Маленький Learning Rate для сохранения предобученных фич бэкбона
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            is_fine_tuning = True

        # --- Цикл обучения ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # --- Цикл валидации ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Обновляем планировщик (только на 2-й стадии)
        if is_fine_tuning and scheduler is not None:
            scheduler.step(acc)
        
        # Сохранение лучшей модели
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), TARGET_DIR / 'best_resnet_model.pth')
            print(f"--> Сохранена новая лучшая модель с точностью: {best_acc:.2f}%")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)

        current_lr = optimizer.param_groups[0]['lr']
        stage_str = "Стадия 1" if not is_fine_tuning else "Стадия 2"
        print(f"[{stage_str}] Эпоха [{epoch+1}/{total_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}% | LR: {current_lr:.6f}")

    # Построение графиков по окончании
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Функция потерь (Loss)')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc', color='green')
    plt.legend()
    plt.title(f'Best Accuracy: {best_acc:.2f}%')
    
    plt.show()

# Запуск пайплайна обучения
if __name__ == "__main__":
    # 30 эпох всего: первые 5 эпох — только классификатор, затем 25 эпох — вся сеть целиком
    train_two_stages(model, train_loader, val_loader, total_epochs=30, freeze_epochs=5)