from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("best.pt")  # Load the YOLOv8 model
results = model.track(source="test2.mp4", conf=0.3, tracker="bytetrack.yaml", persist=True, iou=0.4, show=True)
