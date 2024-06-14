import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from skimage.metrics import structural_similarity as ssim

# Load YOLOv7 model
model_path = 'yolov7.pt'  # Adjust the path to your YOLOv7 model file
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load(model_path, map_location=device)
model.eval()

# Open the live stream
video_path = 'rtsp://admin:Passw0rd!@192.168.227.236:554/live/c443dcad-fa4d-41d7-9de9-9b048aafa823'
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

# Increase buffer size
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# Check if the video stream was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video stream {video_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # Assuming a default FPS if the FPS is not available

# Initialize variables
total_frames = 0
first_frame = None
block_duration = 60  # Duration of each block in seconds
similarity_threshold = 75  # Similarity threshold in percentage
frames_in_block = []

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

# Function to calculate the similarity percentage between two frames
def calculate_similarity(frame1, frame2, mask=None):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    if mask is not None:
        frame1_gray = cv2.bitwise_and(frame1_gray, frame1_gray, mask=mask)
        frame2_gray = cv2.bitwise_and(frame2_gray, frame2_gray, mask=mask)

    score, _ = ssim(frame1_gray, frame2_gray, full=True)
    return score * 100

# Process each frame from the live stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from stream. Skipping to next frame.")
        continue

    total_frames += 1
    current_time = total_frames / fps

    # Add the frame to the current block
    frames_in_block.append(frame)

    # Process blocks every 1 minute
    if current_time >= block_duration:
        block_index = int(current_time // block_duration)

        # Set the first frame as the reference frame for similarity calculation
        if first_frame is None:
            first_frame = frames_in_block[0]

        # Create a cumulative mask for all frames in the block
        cumulative_mask = np.ones(first_frame.shape[:2], dtype=np.uint8) * 255  # Start with a white mask

        for idx, block_frame in enumerate(frames_in_block):
            # Detect objects and create a mask for the current frame
            detections, img_shape = detect_objects(block_frame)
            frame_mask = create_mask(block_frame, detections, img_shape)

            # Update the cumulative mask to include all detected objects
            cumulative_mask = cv2.bitwise_and(cumulative_mask, frame_mask)

        # Calculate similarity for all frames in the current block using the cumulative mask
        all_below_threshold = True
        for idx, block_frame in enumerate(frames_in_block):
            similarity_percentage = calculate_similarity(first_frame, block_frame, cumulative_mask)
            print(f"Block {block_index}, Block Frame {idx + 1}, Total Frame {total_frames}: Similarity {similarity_percentage:.2f}%")
            if similarity_percentage >= similarity_threshold:
                all_below_threshold = False
                break

        # Check the similarity results for the block
        if all_below_threshold:
            print(f"Block {block_index}: Camera angle likely changed.")
        else:
            print(f"Block {block_index}: Camera angle has not changed.")

        # Clear the frames for the next block
        frames_in_block = []
        first_frame = None

        # Update block duration
        block_duration += 60

    # Display the frame (optional, for visualization purposes)
    cv2.imshow('Video Stream', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
