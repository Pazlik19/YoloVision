from dataclasses import dataclass, field
from collections import Counter
import statistics
import math

@dataclass
class Letter:
    # Обязательные поля (передаются при создании)
    id: int
    
    # Списки для хранения истории измерений (используем default_factory!)
    labels: list[str] = field(default_factory=list)
    x_vals: list[float] = field(default_factory=list)
    y_vals: list[float] = field(default_factory=list)
    angles: list[float] = field(default_factory=list)
    
    # Текущие (усредненные) значения
    label: str = ""
    x_mid: float = 0.0
    y_mid: float = 0.0
    angle_mid: float = 0.0
    
    # Стандартное отклонение (погрешность)
    std_x: float = 0.0
    std_y: float = 0.0
    std_angle: float = 0.0
    
    @property
    def count(self) -> int:
        return len(self.x_vals)

    def update(self, label: str, x: float, y: float, angle: float, calcDesp :bool = 0):
        # 1. Добавляем новые измерения
        self.labels.append(label)
        self.x_vals.append(x)
        self.y_vals.append(y)
        self.angles.append(angle)
        
        # 2. Обновляем метку (берем самую частую)
        self.label = Counter(self.labels).most_common(1)[0][0]
        
        # 3. Обновляем средние значения
        self.x_mid = statistics.mean(self.x_vals)
        self.y_mid = statistics.mean(self.y_vals)
        self.angle_mid = statistics.mean(self.angles)
        
        # 4. Обновляем дисперсию (стандартное отклонение), если есть хотя бы 2 замера
        if (self.count > 1 ) and calcDesp:
            self.std_x = statistics.stdev(self.x_vals)
            self.std_y = statistics.stdev(self.y_vals)
            self.std_angle = statistics.stdev(self.angles)



class Scene:
    def __init__(self, distance_threshold: float = 50.0):
        # distance_threshold — радиус, в пределах которого мы узнаем объект
        self.objects: dict[int, Letter] = {}
        self.next_id = 0
        self.distance_threshold = distance_threshold

    def _get_distance(self, x1, y1, x2, y2):
        # Считаем обычное расстояние между двумя точками
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def handle_detection(self, x: float, y: float, angle: float, label: str):
        best_match_id = None
        min_dist = float('inf')

        # 1. Пытаемся найти ближайший существующий объект
        for obj_id, obj in self.objects.items():
            dist = self._get_distance(x, y, obj.x_mid, obj.y_mid)
            if dist < min_dist:
                min_dist = dist
                best_match_id = obj_id

        # 2. Проверяем, укладывается ли расстояние в порог
        if best_match_id is not None and min_dist < self.distance_threshold:
            # Обновляем старый объект
            self.objects[best_match_id].update(label, x, y, angle)
            print(f"Объект {best_match_id} обновлен (дистанция: {min_dist:.2f})")
        else:
            # Создаем новый
            new_letter = Letter(id=self.next_id)
            new_letter.update(label, x, y, angle)
            self.objects[self.next_id] = new_letter
            print(f"Создан новый объект {self.next_id} ({label})")
            self.next_id += 1

    def get_word(self):
        # Собираем слово, сортируя объекты по их координате X (слева направо)
        sorted_objs = sorted(self.objects.values(), key=lambda obj: obj.x_mid)
        return "".join([obj.label for obj in sorted_objs])