from ultralytics import YOLO
model=YOLO('yolo11n.pt')
results=model(source='/home/aryan/8Xengineers/bus.png',show=False, save=True)
# results is a list; get the first result
result = results[0]
print(f'detected {len(result.boxes)} boxes')