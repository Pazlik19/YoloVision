from ultralytics import YOLO

def main():
    # Загружаем предобученную модель YOLOv8 Nano (или yolo11n.pt)
    model = YOLO("yolo11n.pt") 

    # Запускаем обучение
    results = model.train(
        data="yolo_dataset_sam3/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=2,
        device=0, # 0 для видеокарты, или 'cpu', если видеокарты нет
        name="cube_detector", # Имя папки, куда сохранятся результаты
        plots=True # Сохранить графики обучения
    )

if __name__ == '__main__':
    main()