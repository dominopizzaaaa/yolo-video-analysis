import torch
import cv2
import numpy as np
from PIL import Image
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
import os

# Load YOLOv7 model
weights_path = 'yolov7.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path, map_location=device)
model.eval()

# Print the list of classes the model can detect
print("Classes the model can detect:")
print(model.names)

# Load image
image_path = r"C:\Users\Enpag\OneDrive\Desktop\yolo\yolov7-main\videos02\frame_1006.jpg"  # Update this with your image path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

img = Image.open(image_path).convert('RGB')
img = np.array(img)

# Preprocess the image
img_resized = cv2.resize(img, (640, 640))  # Resize image to model input size
img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img_resized = np.ascontiguousarray(img_resized)

# Convert to tensor
img_tensor = torch.from_numpy(img_resized).float().div(255.0).unsqueeze(0).to(device)

# Perform object detection
with torch.no_grad():
    pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

# Process detections
for i, det in enumerate(pred):  # detections per image
    if len(det):
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
        for *xyxy, conf, cls in det:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Save the result image
result_image_path = 'result_image.jpg'
cv2.imwrite(result_image_path, img)

# Display the result image
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
