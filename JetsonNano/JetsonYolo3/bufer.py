from dataclasses import dataclass, field
from collections import Counter
import statistics
import math
from typing import List, Dict

@dataclass
class Letter:
    id: int
    labels: List[str] = field(default_factory=list)
    x_vals: List[float] = field(default_factory=list)
    y_vals: List[float] = field(default_factory=list)
    angles: List[float] = field(default_factory=list)
    
    label: str = ""
    x_mid: float = 0.0
    y_mid: float = 0.0
    angle_mid: float = 0.0
    
    hit_count: int = 0   
    miss_count: int = 0  

    @property
    def count(self) -> int:
        return len(self.x_vals)

    def update(self, label: str, x: float, y: float, angle: float):
        self.labels.append(label)
        self.x_vals.append(x)
        self.y_vals.append(y)
        self.angles.append(angle)
        
        self.label = Counter(self.labels).most_common(1)[0][0]
        self.x_mid = statistics.mean(self.x_vals)
        self.y_mid = statistics.mean(self.y_vals)
        self.angle_mid = statistics.mean(self.angles)
        self.miss_count = 0  
        self.hit_count += 1


class Scene:
    def __init__(self, distance_threshold: float = 5.0, max_misses: int = 15, min_hits: int = 4):
        self.objects: Dict[int, Letter] = {}
        self.next_id = 0
        self.distance_threshold = distance_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        
        # Физический охват камеры (в мм). Настрой под свою оптику!
        self.fov_width = 300.0  
        self.fov_height = 200.0 

    def _get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _is_in_fov(self, obj_x: float, obj_y: float, cam_x: float, cam_y: float) -> bool:
        """
        Проверяет, попадают ли глобальные координаты объекта в текущий объектив камеры.
        Предполагается, что cam_x, cam_y — это центр кадра.
        """
        half_w = self.fov_width / 2
        half_h = self.fov_height / 2
        
        # Оставляем небольшой запас (margin), чтобы объекты на самом краю кадра не "мигали"
        margin = 20.0 
        
        in_x = (cam_x - half_w + margin) <= obj_x <= (cam_x + half_w - margin)
        in_y = (cam_y - half_h + margin) <= obj_y <= (cam_y + half_h - margin)
        
        return in_x and in_y

    def update_scene(self, current_detections: list, cam_x: float = 0.0, cam_y: float = 0.0):
        matched_ids = set()

        # 1. Ассоциация (перевод локальных координат в глобальные)
        for det in current_detections:
            global_x = cam_x + det["x"]
            global_y = cam_y + det["y"]
            
            best_match_id = None
            min_dist = float('inf')

            for obj_id, obj in self.objects.items():
                dist = self._get_distance(global_x, global_y, obj.x_mid, obj.y_mid)
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = obj_id

            if best_match_id is not None and min_dist < self.distance_threshold:
                self.objects[best_match_id].update(det["label"], global_x, global_y, det["angle"])
                matched_ids.add(best_match_id)
            else:
                new_letter = Letter(id=self.next_id)
                new_letter.update(det["label"], global_x, global_y, det["angle"])
                self.objects[self.next_id] = new_letter
                matched_ids.add(self.next_id)
                self.next_id += 1

        # 2. Умная очистка (наказываем только тех, кто перед "глазами", но не детектируется)
        for obj_id, obj in list(self.objects.items()):
            if obj_id not in matched_ids:
                if self._is_in_fov(obj.x_mid, obj.y_mid, cam_x, cam_y):
                    # Объект должен быть в кадре, но YOLO его не прислал
                    obj.miss_count += 1
                    if obj.miss_count >= self.max_misses:
                        print(f"[BUFFER] Ложный/исчезнувший объект {obj_id} удален из памяти.")
                        del self.objects[obj_id]
                else:
                    # Объект вне зоны видимости (камера уехала). Ничего не делаем!
                    # Он безопасно хранится в глобальной карте.
                    pass

    def get_all_objects_data(self) -> list:
        objects_list = []
        confirmed_objs = [obj for obj in self.objects.values() if obj.hit_count >= self.min_hits]
        sorted_objs = sorted(confirmed_objs, key=lambda obj: obj.x_mid)
        
        for obj in sorted_objs:
            objects_list.append({
                "id": obj.id,
                "label": obj.label,
                "x": round(obj.x_mid, 2),
                "y": round(obj.y_mid, 2),
                "angle": round(obj.angle_mid, 2),
                "hit_count": obj.hit_count
            })
        return objects_list

    def get_word(self) -> str:
        confirmed_objs = [obj for obj in self.objects.values() if obj.hit_count >= self.min_hits]
        sorted_objs = sorted(confirmed_objs, key=lambda obj: obj.x_mid)
        return "".join([obj.label for obj in sorted_objs])