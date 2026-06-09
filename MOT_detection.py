import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
vehicle_classes = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

video_path = "/home/aryan/comvis/traffic_video.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Video not found"

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

window_name = "YOLOv11 detections"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
max_width = 1280

while True:
	ret, frame = cap.read()
	if not ret:
		break

	height, width = frame.shape[:2]
	if width > max_width:
		scale = max_width / width
		frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

	results = model(frame, verbose=False)[0]
	detections = sv.Detections.from_ultralytics(results)

	class_ids = np.array(detections.class_id, dtype=int)
	detections = detections[(detections.confidence > 0.4) & np.isin(class_ids, list(vehicle_classes))]

	labels = [f"{results.names[int(class_id)]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]

	frame = box_annotator.annotate(frame, detections)
	frame = label_annotator.annotate(frame, detections, labels=labels)
	cv2.putText(frame, "Press q to quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

	cv2.imshow(window_name, frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()