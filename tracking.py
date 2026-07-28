import cv2 as cv
import numpy as np
from ultralytics import YOLO
from boxmot.trackers import ByteTrack

colors = np.random.randint(0, 255, size=(1000, 3), dtype="uint8")


def get_color(track_id):
    return tuple(int(c) for c in colors[int(track_id) % 1000])


def draw_track(frame, x1, y1, x2, y2, track_id, cls_name, conf):
    color = get_color(track_id)
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id} | {cls_name} {conf:.2f}"
    (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv.putText(frame, label, (x1, y1 - 4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


video_path = "/home/aryan/comvis/traffic_video.mp4"
output_path = "/home/aryan/comvis/output_tracked.mp4"
model = YOLO("yolo11m.pt")
tracker = ByteTrack(
    track_thresh=0.3,
    match_thresh=0.8,
    track_buffer=30
)
cap = cv.VideoCapture(video_path)
assert cap.isOpened(), f"Video not found: {video_path}"

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

writer = cv.VideoWriter(
    output_path,
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print(" GPU start")
for _ in range(5):
    ret, frame = cap.read()
    if ret:
        model(frame, verbose=False)

cap.set(cv.CAP_PROP_POS_FRAMES, 0)

import time

frame_count = 0
start_time = time.time()
max_id = 0

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]

    dets_np = result.boxes.data.cpu().numpy()

    tracks = tracker.update(dets_np, frame)

    if len(tracks) > 0:
        max_id = max(max_id, int(np.max(tracks[:, 4])))

        for track in tracks:
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            track_id = int(track[4])
            conf = float(track[5])
            cls_id = int(track[6])
            cls_name = result.names[cls_id]

            frame = draw_track(frame, x1, y1, x2, y2, track_id, cls_name, conf)

    elapsed = time.time() - start_time
    live_fps = frame_count / elapsed if elapsed > 0 else 0
    n_tracks = len(tracks)

    cv.putText(frame, f"FPS: {live_fps:.1f}  Tracks: {n_tracks}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    writer.write(frame)
    frame_count += 1

    if frame_count % 50 == 0:
        print(f"  Frame {frame_count} | FPS: {live_fps:.1f} | Active tracks: {n_tracks}")

print(f"Highest track ID assigned: {max_id}")
cap.release()
writer.release()
total_time = time.time() - start_time
print(f"\nDone.")
print(f"  Total frames : {frame_count}")
print(f"  Average FPS  : {frame_count / total_time:.1f}")
print(f"  Output saved : {output_path}")