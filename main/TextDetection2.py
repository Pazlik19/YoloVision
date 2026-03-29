import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import json
import numpy as np
import cv2
import os
from pathlib import Path
MODEL_DIR = Path(__file__).parent.parent / "Models"
class SymbolClassifier:
    def __init__(self, model_path= MODEL_DIR/'best_resnet_model.pth',  config_path= MODEL_DIR/'class_names.json'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфиг не найден: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализируем ResNet18 (как при обучении)
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Веса не найдены: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        # Трансформации строго как при обучении ResNet
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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