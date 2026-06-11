import cv2 as cv
import supervision as sv
from ultralytics import YOLO
import time

video_path = "/home/aryan/comvis/traffic_video.mp4"

def benchmark(model_name, video_path, max_frames=100):
    model = YOLO(model_name)
    cap = cv.VideoCapture(video_path)

    # warmup — run 5 frames before timing
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            model(frame, verbose=False)

    total_dets = 0
    start = time.time()

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        result = model(frame, verbose=False)[0]
        detect = sv.Detections.from_ultralytics(result)
        detect = detect[detect.confidence > 0.3]
        total_dets += len(detect)

    elapsed = time.time() - start
    fps = max_frames / elapsed

    cap.release()
    print(f"\n{model_name}")
    print(f"  FPS:              {fps:.1f}")
    print(f"  Avg detections:   {total_dets / max_frames:.1f} per frame")

benchmark("yolo11n.pt", video_path)
benchmark("yolo11m.pt", video_path)   # downloads ~40MB first run
benchmark("rtdetr-l.pt", video_path)  # downloads ~120MB first run