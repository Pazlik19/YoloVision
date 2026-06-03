import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import json
import copy
from pathlib import Path 
TARGET_DIR = Path(__file__).parent.parent / "Models"

# 1. Трансформации (оставляем как в прошлый раз, они рабочие)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(90), 
    transforms.ColorJitter(
        brightness=0.3, # Меняем яркость
        contrast=0.3    # Меняем контрастность, чтобы выделить/скрыть рельеф
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Данные
full_dataset = datasets.ImageFolder(root='data/train/', transform=transform)
with open(TARGET_DIR /'class_names.json', 'w', encoding='utf-8') as f:
    json.dump(full_dataset.classes, f)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # Уменьшили батч для стабильности
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. Модель: Размораживаем ВСЁ для тонкой настройки
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))

# Включаем градиенты для всех параметров
for param in model.parameters():
    param.requires_grad = True

# 4. Улучшенная функция обучения
def train_fine_tuning(model, train_loader, val_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    
    criterion = nn.CrossEntropyLoss()
    # Маленький LR, чтобы не "взорвать" предобученные веса
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    # Планировщик: если 5 эпох точность не растет, уменьшаем LR в 2 раза
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
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
        
        # Валидация
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
        
        # Обновляем планировщик по точности
        scheduler.step(acc)
        
        # Сохраняем веса, если точность лучшая
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, TARGET_DIR/'best_resnet_model.pth')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Визуализация (сокращено)
    plt.plot(history['val_acc'], label='Acc')
    plt.title(f'Best Accuracy: {best_acc:.2f}%')
    plt.show()

# Запуск
train_fine_tuning(model, train_loader, val_loader, epochs=30)