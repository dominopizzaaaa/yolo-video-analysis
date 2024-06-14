import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms

# YOLOv7 setup
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Suppress FFmpeg warnings and errors
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
sys.stderr = open(os.devnull, 'w')

# Load YOLOv7 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('yolov7.pt', map_location=device)
model.eval()
print("YOLOv7 model loaded")

# Function to detect objects using YOLOv7
def detect_objects(frame, model, device):
    img_size = 640
    img = letterbox(frame, img_size, stride=32, auto=True)[0]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)[0]

    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    det = pred[0]
    return det

# Path to the video file
video_path = 'C:\\Users\\Enpag\\OneDrive\\Desktop\\yolo\\yolov7-main\\inference\\videos\\cctv.mp4'
cap = cv2.VideoCapture(video_path)
print("Video capture initialized")

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()
print("Video file opened successfully")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # Assuming a default FPS if the FPS is not available
print(f"Video FPS: {fps}")

# Initialize variables
total_frames = 0
similarity_threshold = 90  # Similarity threshold in percentage
first_frame = None

# Function to calculate the similarity percentage between two frames
def calculate_similarity(frame1, frame2, mask=None):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    if mask is not None:
        frame1_gray = cv2.bitwise_and(frame1_gray, frame1_gray, mask=mask)
        frame2_gray = cv2.bitwise_and(frame2_gray, frame2_gray, mask=mask)
    
    score, _ = ssim(frame1_gray, frame2_gray, full=True)
    return score * 100

# Process each frame
while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video.")
            break

        total_frames += 1
        print(f"Processing frame {total_frames}")

        if first_frame is None:
            first_frame = frame  # Store the first frame as the reference frame
            print("First frame set as reference frame")
            continue

        # Detect objects in the current frame
        print("Starting detection")
        det = detect_objects(frame, model, device)
        print(f"Detection completed for frame {total_frames}")
        
        # Create a mask for detected objects
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255  # Start with a white mask
        if det is not None and len(det):
            print(f"Detected {len(det)} objects in frame {total_frames}")
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                mask[y1:y2, x1:x2] = 0  # Black out detected object area

        # Calculate similarity for every frame after the first frame
        print("Starting similarity calculation")
        similarity_percentage = calculate_similarity(first_frame, frame, mask=mask)
        print(f"Frame {total_frames}: Similarity {similarity_percentage:.2f}%")
        if similarity_percentage < similarity_threshold:
            print("Camera angle likely changed.")
        else:
            print("Camera angle has not changed.")

        cv2.imshow('Video Stream', frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

# Release resources
cap.release()
cv2.destroyAllWindows()

# Restore standard error
sys.stderr = sys._stderr_

print("Script finished")
