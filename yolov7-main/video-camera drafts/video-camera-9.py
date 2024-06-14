import cv2
import time
import os
import numpy as np
import torch
import sys
import threading
from datetime import datetime, timedelta
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
model_path = 'yolov7.pt'  # Path to my YOLOv7 model file
device = 'cuda' if torch.cuda.is_available() else 'cpu' # check if CUDA and GPU available on computer, else use CPU
model = attempt_load(model_path, map_location=device) # loads a pre-trained model from YOLO to the CPU or GPU
model.eval()

# Function to detect objects using YOLOv7
def detect_objects(frame):
    img = cv2.resize(frame, (640, 640))  # Resize frame to 640x640
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW (required for deep learning models)
    img = np.ascontiguousarray(img)  # Ensure memory is contiguous for efficient processing
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # Convert to torch tensor

    img = img.to(device) # moves image tensor to CPU / GPU
    with torch.no_grad(): # don't track gradient for efficiency, obtaining the predictions 
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
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame from BGR to grayscale.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create a CLAHE object with specified clip limit and tile grid size.
    frame_equalized = clahe.apply(frame_gray)  # Apply CLAHE to the grayscale frame to enhance contrast.
    return frame_equalized  # Return the contrast-enhanced grayscale frame.

# Function to calculate the similarity percentage between two frames using SSI Mby first normalizing their lighting conditions, optionally applying a mask
def calculate_similarity(frame1, frame2, mask=None):
    frame1_gray = normalize_lighting(frame1)
    frame2_gray = normalize_lighting(frame2)
    
    if mask is not None:
        frame1_gray = cv2.bitwise_and(frame1_gray, frame1_gray, mask=mask)
        frame2_gray = cv2.bitwise_and(frame2_gray, frame2_gray, mask=mask)

    score, diff = ssim(frame1_gray, frame2_gray, full=True)
    return score * 100, diff

# Function to load the reference frame based on the current timestamp
def load_reference_frame(output_dir, current_time_str):
    reference_frame_path = os.path.join(output_dir, f"frame_{current_time_str}.jpg")
    if os.path.exists(reference_frame_path):
        return cv2.imread(reference_frame_path), reference_frame_path
    else:
        return None, None

# Function to list all reference times from filenames in the directory
def list_reference_times(output_dir):
    reference_times = []
    for filename in os.listdir(output_dir):
        if filename.startswith("frame_") and filename.endswith(".jpg"):
            time_str = filename[6:-4]  # Extract the time string from the filename
            reference_times.append(time_str)
    return sorted(reference_times)

# Function to capture frames from an RTSP stream and compare with reference frames
def capture_rtsp_frames(rtsp_url, reference_dir):
    reference_times = list_reference_times(reference_dir)
    if not reference_times:
        print("No reference frames found in the directory.")
        return
    
    while True:
        current_time = datetime.now().replace(second=0, microsecond=0)
        current_time_str = current_time.strftime("%H%M")
        
        print(f"Current time: {current_time_str}")
        
        if current_time_str in reference_times:
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

            print(f"Captured frame at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            reference_frame, reference_frame_path = load_reference_frame(reference_dir, current_time_str)  # Load the reference frame from the specified directory
            
            if reference_frame is not None:
                print(f"Using reference frame: {reference_frame_path}")
                # Process the captured frame for analysis
                try:
                    process_frame(frame, current_time, reference_frame, similarity_threshold)
                except Exception as e:
                    print(f"Error processing frame: {e}")
            else:
                print(f"No matching reference frame found for time {current_time_str}")

        time.sleep(60)

# Function to process each captured frame for analysis
def process_frame(frame, current_time, reference_frame, similarity_threshold):
    # Detect objects and create a mask for the current frame
    detections, img_shape = detect_objects(frame)
    frame_mask = create_mask(frame, detections, img_shape)

    # Calculate similarity with the reference frame using the mask
    similarity_percentage, diff = calculate_similarity(reference_frame, frame, frame_mask)
    
    # Get the current time and date
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    if similarity_percentage < similarity_threshold:
        print(f"{current_time_str}: Similarity {similarity_percentage:.2f}% - Camera angle likely changed.")
    else:
        print(f"{current_time_str}: Similarity {similarity_percentage:.2f}% - No change")

# Global parameters
similarity_threshold = 60  # Similarity threshold in percentage

# Main function to handle inputs and start capturing
def main():
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("Usage: python video-camera-9.py <rtsp_url1> <reference_directory1> [<rtsp_url2> <reference_directory2> ...]")
        return

    threads = []
    for i in range(1, len(sys.argv), 2):
        rtsp_url = sys.argv[i]
        reference_dir = sys.argv[i + 1]
        
        thread = threading.Thread(target=capture_rtsp_frames, args=(rtsp_url, reference_dir))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()

