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
video_path = r'C:\Users\Enpag\OneDrive\Desktop\yolo\yolov7-main\inference\videos\sudden-change.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Initialize variables
total_frames = 0
changed_frames = 0
prev_objects = None
threshold_percentage = 5  # Initial threshold percentage
buffer_percentage = 2  # Initial buffer percentage
min_change_frames = 10  # Minimum number of frames to calculate adaptive thresholds
change_ratio_history = []

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

# Function to calculate the overlap of non-living objects
def calculate_non_living_overlap(objects):
    non_living_classes = [0, 1, 2, 3, 5, 7]  # Assuming class indices for non-living objects
    non_living_objects = []
    
    for obj in objects:
        for o in obj:
            if int(o[5]) in non_living_classes:
                non_living_objects.append(o)
    
    if not non_living_objects:
        return 0

    # Calculate the area covered by non-living objects
    total_area = sum([(obj[3] - obj[1]) * (obj[2] - obj[0]) for obj in non_living_objects])
    return total_area

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # Detect objects in the current frame
    objects = detect_objects(frame)
    
    if prev_objects is not None:
        # Calculate overlap between current and previous frame
        prev_non_living_area = calculate_non_living_overlap(prev_objects)
        curr_non_living_area = calculate_non_living_overlap(objects)

        # Calculate change ratio and update history
        if prev_non_living_area > 0:  # Prevent division by zero
            change_ratio = abs(curr_non_living_area - prev_non_living_area) / prev_non_living_area
            change_ratio_history.append(change_ratio)

            # Adaptive thresholding
            if len(change_ratio_history) >= min_change_frames:
                # Calculate adaptive threshold based on historical change ratios
                mean_change_ratio = np.mean([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in change_ratio_history])
                threshold_percentage = mean_change_ratio * 100

            # Check if the change is significant
            if change_ratio > (threshold_percentage + buffer_percentage) / 100:
                changed_frames += 1
                print(f"Frame {total_frames}: Camera angle has changed.")
            else:
                print(f"Frame {total_frames}: Camera angle has not changed.")
        else:
            print(f"Frame {total_frames}: Previous non-living area is zero, cannot calculate change ratio.")
    
    # Update previous objects for the next iteration
    prev_objects = objects

    # Clear CUDA cache to free up memory
    if total_frames % 10 == 0:  # Clear cache every 10 frames
        torch.cuda.empty_cache()

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate the percentage of frames with significant changes
if total_frames > 0:
    percentage_changed = (changed_frames / total_frames) * 100
    print(f"Total frames: {total_frames}")
    print(f"Frames with significant camera angle change: {changed_frames}")
    print(f"Percentage of frames with camera angle change: {percentage_changed:.2f}%")
else:
    print("No frames processed.")

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()