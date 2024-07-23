from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('videos/murraygw.mp4', save=True)

print(results[0])