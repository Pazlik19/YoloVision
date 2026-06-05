import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_multiple_texts(image, texts_with_positions, font_path, font_size=24, color=(0, 255, 0)):
    """Рисует весь текст на кадре за одну конвертацию в PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        font = ImageFont.load_default()
        
    for text, position in texts_with_positions:
        # stroke_width даёт тёмную обводку: мелкий текст остаётся читаемым
        # и на светлом столе, и на оранжевой грани кубика.
        draw.text(position, text, font=font, fill=color,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)