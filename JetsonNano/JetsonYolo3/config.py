import os
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR.parent / "Models"

# ZMQ Настройки
ZMQ_ADDRESS = "tcp://*:5555"

# Физические параметры сцены (в мм)
CAMERA_HEIGHT_MM = 200 

# Настройка шрифтов (автовыбор OS)
FONT_PATH_LINUX = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
FONT_PATH_WINDOWS = "C:/Windows/Fonts/arial.ttf" 
DEFAULT_FONT_PATH = FONT_PATH_WINDOWS if os.name == 'nt' else FONT_PATH_LINUX