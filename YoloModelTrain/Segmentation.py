from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path 
TARGET_DIR = Path(__file__).parent.parent / "Models" 
class Segmentation:
    def __init__(self, model_path=TARGET_DIR /'best.pt'):
        # Загружаем предобученную модель YOLO (segmentation)
        self.model = YOLO(model_path)

    def get_objects(self, image):
        """
        Находит объекты, вырезает их по маске и возвращает:
        (список вырезанных картинок, список координат BBox)
        """
        results = self.model(image, verbose=False)
        crops = []
        bboxes = []
        
        if not results[0].masks:
            return crops, bboxes

        # Оригинальное изображение
        img_orig = results[0].orig_img
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]

        for i, mask in enumerate(masks):
            # Масштабируем маску под размер оригинала
            mask_resized = cv2.resize(mask, (img_orig.shape[1], img_orig.shape[0]))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            
            x1, y1, x2, y2 = map(int, boxes[i])
            
            # Вырезаем область
            crop = img_orig[y1:y2, x1:x2]
            crop_mask = binary_mask[y1:y2, x1:x2]
            
            # Накладываем маску (черный фон вокруг объекта)
            final_crop = cv2.bitwise_and(crop, crop, mask=crop_mask)
            
            crops.append(final_crop)
            bboxes.append((x1, y1, x2, y2))
            
        return crops, bboxes