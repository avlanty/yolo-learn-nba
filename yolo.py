from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict('videos/murraygw.mp4', save=True)
print(results[0])