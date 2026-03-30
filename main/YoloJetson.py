import cv2
import os
from pathlib import Path
# ... твои остальные импорты (YOLO, и т.д.)

# 1. Перед началом цикла (после инициализации камеры и модели)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Указываем имя файла, кодек, FPS (поставь 5-10) и размер кадра
out = cv2.VideoWriter('output_video.avi', fourcc, 8.0, (640, 480))

print("Запись пошла... Покрути кубики перед камерой секунд 10, потом нажми 'q'")

try:
    while True:
        # ... твой код захвата кадра и работы YOLO ...
        # ret, frame = cap.read()
        # results = detector(frame)
        
        # 2. Вместо (или вместе) с cv2.imshow:
        # cv2.imshow("YOLO + ResNet Classification", frame) # Эту строку можно закомментировать
        
        out.write(frame) # ЗАПИСЫВАЕМ КАДР В ФАЙЛ
        
        # Для выхода из цикла в докере без графики 
        # лучше использовать ограничение по кадрам или Ctrl+C
        # но если хочешь по нажатию клавиши (в консоли), оставь:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Запись остановлена пользователем")

finally:
    # 3. ОБЯЗАТЕЛЬНО закрываем запись, иначе файл не откроется!
    out.release()
    # cap.release()
    print("Готово! Файл 'output_video.avi' сохранен в папке проекта.")