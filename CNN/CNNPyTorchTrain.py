import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
# 1. Настройка трансформаций (Препроцессинг)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    
    # --- МОЩНАЯ АУГМЕНТАЦИЯ ---
    
    # Случайный поворот до 30 градусов и небольшие сдвиги (перспективные искажения)
    transforms.RandomAffine(
        degrees=90, 
        translate=(0.1, 0.1), # Сдвиг по горизонтали/вертикали на 10%
        scale=(0.8, 1.2),     # Масштабирование (зум) от 80% до 120%
        shear=10              # Сдвиг (наклон)
    ),
    
    # Имитация разного освещения (очень важно для теней букв)
    transforms.ColorJitter(
        brightness=0.3, # Меняем яркость
        contrast=0.3    # Меняем контрастность, чтобы выделить/скрыть рельеф
    ),
    
    # Добавим немного случайного размытия (имитация расфокуса камеры)
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.2), # С вероятностью 20%
    
    # --------------------------

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomInvert(p=0.5)
])

# 2. Загрузка данных (предполагаем, что фото лежат в папке 'data/train/буква_А/...')
train_dataset = datasets.ImageFolder(root='data/train/', transform=transform)
# Посмотрим, как распределились классы
print(f"Словарь классов: {train_dataset.class_to_idx}")

# Сохраним список имен в файл, чтобы потом использовать в скрипте предсказания

with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(train_dataset.classes, f)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Архитектура нейросети
class SymbolCNN(nn.Module):
    def __init__(self, num_classes):
        super(SymbolCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 4. Цикл обучения (Training Loop)
def train_model(model, train_loader, epochs=3000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loss_history = []  # Список для хранения потерь
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0: # Печатаем каждые 10 эпох, чтобы не спамить
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            loss_history.append(epoch_loss) # Сохраняем средний лосс за эпоху

    # --- БЛОК ПОСТРОЕНИЯ ГРАФИКА ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('График функции потерь (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png') # Сохраняем график в файл
    plt.show() # Показываем окно с графиком
    # ------------------------------

    torch.save(model.state_dict(), 'symbol_model.pth')
    print("Модель сохранена и график построен!")

# Запуск
model = SymbolCNN(num_classes=len(train_dataset.classes))
train_model(model, train_loader)
