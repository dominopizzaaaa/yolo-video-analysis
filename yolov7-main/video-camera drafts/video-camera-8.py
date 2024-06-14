import cv2
import time
import os
import numpy as np
import torch
import sys
from datetime import datetime
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from skimage.metrics import structural_similarity as ssim

# Redirect stderr to suppress FFmpeg warnings
class SuppressStream:
    def __init__(self, stream):
        self.stream = stream
        self.null_stream = open(os.devnull, 'w')
        
    def __enter__(self):
        self.original_stream = os.dup(self.stream.fileno())
        os.dup2(self.null_stream.fileno(), self.stream.fileno())
        
    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.original_stream, self.stream.fileno())
        self.null_stream.close()

# Load YOLOv7 model
model_path = 'yolov7.pt'  # Adjust the path to your YOLOv7 model file
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load(model_path, map_location=device)
model.eval()

# Function to detect objects using YOLOv7
def detect_objects(frame):
    img = cv2.resize(frame, (640, 640))  # Resize frame to 640x640
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)  # Ensure memory is contiguous
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # Convert to torch tensor

    img = img.to(device)
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)  # Adjust confidence threshold as needed

    return pred, img.shape

# Function to create a mask from detected objects
def create_mask(frame, detections, img_shape):
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255  # Start with a white mask
    for det in detections:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det.cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                mask[y1:y2, x1:x2] = 0  # Black out detected object area
    return mask

# Function to normalize the lighting conditions of an image using CLAHE
def normalize_lighting(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_equalized = clahe.apply(frame_gray)
    return frame_equalized

# Function to detect the sky area and calculate its brightness
def calculate_sky_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_sky = np.array([0, 0, 50])  # Adjust the lower threshold for sky detection
    upper_sky = np.array([180, 50, 255])  # Adjust the upper threshold for sky detection
    sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
    
    # Calculate the brightness of the sky area
    v_channel = hsv[:, :, 2]
    masked_v_channel = np.ma.masked_array(v_channel, mask=~sky_mask)
    brightness = masked_v_channel.mean()
    
    return brightness

# Function to calculate the similarity percentage between two frames using SSIM
def calculate_similarity(frame1, frame2, mask=None):
    frame1_gray = normalize_lighting(frame1)
    frame2_gray = normalize_lighting(frame2)
    
    if mask is not None:
        frame1_gray = cv2.bitwise_and(frame1_gray, frame1_gray, mask=mask)
        frame2_gray = cv2.bitwise_and(frame2_gray, frame2_gray, mask=mask)

    score, diff = ssim(frame1_gray, frame2_gray, full=True)
    return score * 100, diff

# Function to capture and save frames from an RTSP stream
def capture_rtsp_frames(rtsp_url, output_dir, interval=60, ref_interval=300):
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    reference_frame = None

    while True:
        with SuppressStream(sys.stderr):
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            if not cap.isOpened():
                print(f"Error: Could not open RTSP stream at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Retrying in 5 seconds...")
                cap.release()
                time.sleep(5)
                continue
        
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Captured and saved {frame_path}")

        if frame_count % (ref_interval // interval) == 0:  # Update the reference frame every 5 minutes
            reference_frame = frame
            print(f"Updated reference frame at {frame_path}")

        # Process the captured frame for analysis
        try:
            if reference_frame is not None:
                process_frame(frame, frame_count, reference_frame, similarity_threshold)
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        frame_count += 1
        cap.release()
        time.sleep(interval)

# Function to process each captured frame for analysis
def process_frame(frame, frame_count, reference_frame, similarity_threshold):
    # Calculate sky brightness
    brightness = calculate_sky_brightness(frame)

    # Detect objects and create a mask for the current frame
    detections, img_shape = detect_objects(frame)
    frame_mask = create_mask(frame, detections, img_shape)

    # Calculate similarity with the reference frame using the mask
    similarity_percentage, diff = calculate_similarity(reference_frame, frame, frame_mask)
    
    # Get the current time and date
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if similarity_percentage < similarity_threshold:
        print(f"{current_time} - Frame {frame_count + 1}: Brightness {brightness:.2f} - Similarity {similarity_percentage:.2f}% - Camera angle likely changed.")
    else:
        print(f"{current_time} - Frame {frame_count + 1}: Brightness {brightness:.2f} - Similarity {similarity_percentage:.2f}% - No change")

# Global parameters
similarity_threshold = 60  # Similarity threshold in percentage

# RTSP stream URL
rtsp_url = 'rtsp://service:Passw0rd!@192.168.229.116:554/Streaming/Channels/101'  # Adjust the URL format based on your camera's requirements
output_dir = 'captured_frames'
capture_interval = 600  # Capture a frame every 10 minute
reference_interval = 3600  # Capture a reference frame every 60 minutes

# Start capturing frames from the RTSP stream
capture_rtsp_frames(rtsp_url, output_dir, capture_interval, reference_interval)

print("Processing complete.")
