import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load YOLOv7 model
model_path = 'yolov7.pt'  # Adjust the path to your YOLOv7 model file
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load(model_path, map_location=device)
model.eval()

# Open the video file
video_path = r'C:\Users\Enpag\OneDrive\Desktop\yolo\yolov7-main\inference\videos\slow-change.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Initialize variables
total_frames = 0
sudden_changes = 0
slow_changes = 0
prev_frame = None
prev_objects = None
motion_threshold = 1  # Threshold for detecting significant motion
slow_change_threshold = 0.2  # Threshold for detecting slow change
non_living_classes = [0, 1, 2, 3, 5, 7, 56]  # Assuming class indices for non-living objects and chairs
consecutive_sudden_changes = 0
max_consecutive_sudden_changes = 0

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

    # Convert list of tensors to list of numpy arrays
    pred = [p.cpu().numpy() for p in pred if p is not None]

    return pred

# Function to filter out humans and other moving objects
def filter_non_living_objects(objects):
    non_living_objects = []
    for obj in objects:
        for o in obj:
            if int(o[5]) in non_living_classes:
                non_living_objects.append(o)
    return non_living_objects

# Function to calculate frame difference
def calculate_frame_difference(prev_frame, curr_frame):
    diff = cv2.absdiff(prev_frame, curr_frame)
    non_zero_count = np.count_nonzero(diff)
    return non_zero_count / diff.size

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects in the current frame
    objects = detect_objects(frame)
    non_living_objects = filter_non_living_objects(objects)

    if prev_frame is not None:
        # Calculate frame difference
        frame_diff = calculate_frame_difference(prev_frame, gray_frame)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Check for motion in the scene
        motion_detected = np.mean(mag) > motion_threshold

        if motion_detected:
            sudden_changes += 1
            slow_changes += 1
            consecutive_sudden_changes += 1
            print(f"Frame {total_frames}: Camera angle has changed (sudden change detected).")
        elif frame_diff > slow_change_threshold:
            slow_changes += 1
            consecutive_sudden_changes = 0
            print(f"Frame {total_frames}: Camera angle has changed (slow change detected).")
        else:
            consecutive_sudden_changes = 0
            print(f"Frame {total_frames}: Camera angle has not changed.")

        max_consecutive_sudden_changes = max(max_consecutive_sudden_changes, consecutive_sudden_changes)

    # Update previous frame for the next iteration
    prev_frame = gray_frame

    # Update previous objects for the next iteration
    prev_objects = non_living_objects

    # Clear CUDA cache to free up memory
    if total_frames % 10 == 0:  # Clear cache every 10 frames
        torch.cuda.empty_cache()

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate the probability of camera shift
probability_sudden_shift = (sudden_changes / total_frames) * 100
probability_slow_shift = (slow_changes / total_frames) * 100
print(f"Total frames: {total_frames}")
print(f"Sudden changes detected: {sudden_changes}")
print(f"Slow changes detected: {slow_changes}")
print(f"Probability of sudden shift: {probability_sudden_shift:.2f}%")
print(f"Probability of shift: {probability_slow_shift:.2f}%")

if max_consecutive_sudden_changes > 10:
    print("Likely to have a sudden shift in camera.")

if probability_slow_shift > 70:
    print("Likely to have a slow shift in camera.")

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()