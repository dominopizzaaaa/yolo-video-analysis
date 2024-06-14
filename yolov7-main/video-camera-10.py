import cv2
import time
import os
import numpy as np
import torch
import sys
import redis
import threading
from datetime import datetime, timedelta
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import signal
from scipy.ndimage import laplace

# Set up Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

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

# Function to calculate the similarity percentage between two frames using SSIM
def calculate_similarity(frame1, frame2, mask=None):
    frame1_gray = normalize_lighting(frame1)
    frame2_gray = normalize_lighting(frame2)
    
    if mask is not None:
        frame1_gray = cv2.bitwise_and(frame1_gray, frame1_gray, mask=mask)
        frame2_gray = cv2.bitwise_and(frame2_gray, frame2_gray, mask=mask)

    score, diff = ssim(frame1_gray, frame2_gray, full=True)
    return score * 100, diff

# Function to load the reference frame from Redis
def load_reference_frame_from_redis(key):
    image_data = redis_client.get(key)
    if image_data:
        image_array = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return None

# Function to load the reference frame based on the current timestamp
def load_reference_frame(output_dir, current_time_str):
    reference_frame = load_reference_frame_from_redis(f"frame_{current_time_str}.jpg")
    if reference_frame is not None:
        return reference_frame, f"frame_{current_time_str}.jpg"
    
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
            time_str = filename[6:10]
            reference_times.append(time_str)
    return sorted(reference_times)

# Function to measure blur intensity
def measure_blur_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = laplace(gray)
    blur_intensity = np.var(laplacian)
    return blur_intensity

# Function to detect edges
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Function to analyze edge distribution
def analyze_edge_distribution(edges):
    edge_coords = np.column_stack(np.where(edges > 0))
    if edge_coords.size == 0:
        return 0, [0, 0]
    
    edge_variance = np.var(edge_coords, axis=0)
    edge_density = edge_coords.shape[0] / (edges.shape[0] * edges.shape[1])
    return edge_density, edge_variance

# Function to detect water droplets on the lens
def detect_water_droplets(image, blur_threshold=13010, edge_density_threshold=0.1, edge_variance_thresholds=(40000, 200000)):
    blur_intensity = measure_blur_intensity(image)
    edges = edge_detection(image)
    edge_density, edge_variance = analyze_edge_distribution(edges)
    
    # Debug prints for each condition
    print(f"Blur intensity check: {blur_intensity} > {blur_threshold} = {blur_intensity > blur_threshold}")
    print(f"Edge density check: {edge_density} < {edge_density_threshold} = {edge_density < edge_density_threshold}")
    print(f"Edge variance check: {edge_variance[0]} < {edge_variance_thresholds[0]} and {edge_variance[1]} < {edge_variance_thresholds[1]} = {edge_variance[0] < edge_variance_thresholds[0] and edge_variance[1] < edge_variance_thresholds[1]}")
    
    # Combine multiple heuristics for better detection
    droplets_detected = (
        (blur_intensity > blur_threshold) or
        (edge_density < edge_density_threshold) or
        (edge_variance[0] < edge_variance_thresholds[0] and edge_variance[1] < edge_variance_thresholds[1])
    )
    
    return droplets_detected, blur_intensity, edge_density, edge_variance

# Function to capture frames from an RTSP stream and compare with reference frames
def capture_rtsp_frames(rtsp_url, reference_dir):
    reference_times = list_reference_times(reference_dir)
    if not reference_times:
        print("No reference frames found in the directory.")
        return
    
    timestamps = []
    similarity_scores = []
    changed_count = 0
    changed_images_dir = f"changed_images_{reference_dir}"
    water_droplets_dir = f"water_droplets_{reference_dir}"
    os.makedirs(changed_images_dir, exist_ok=True)
    os.makedirs(water_droplets_dir, exist_ok=True)
    last_print_time = time.time()

    while True:
        current_time = datetime.now().replace(second=0, microsecond=0)
        current_time_str = current_time.strftime("%H%M")
        
        if time.time() - last_print_time >= 600:
            print(f"Current time: {current_time_str}")
            last_print_time = time.time()

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

            droplets_detected, blur_intensity, edge_density, edge_variance = detect_water_droplets(frame)

            if droplets_detected:
                print(f"Water droplets/Dust detected on the lens. Blur intensity: {blur_intensity:.2f}, Edge density: {edge_density:.4f}, Edge variance: {edge_variance}")
                save_path = os.path.join(water_droplets_dir, f"droplet_frame_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"Saved water droplet/dust frame to {save_path}")
                continue

            reference_frame, reference_frame_path = load_reference_frame(reference_dir, current_time_str)  # Load the reference frame from the specified directory
            
            if reference_frame is not None:
                print(f"Using reference frame: {reference_frame_path}")
                # Process the captured frame for analysis
                try:
                    similarity_percentage, camera_angle_changed = process_frame(frame, current_time, reference_frame, similarity_threshold, changed_images_dir, reference_dir)
                    timestamps.append(current_time_str)
                    similarity_scores.append(similarity_percentage)
                    
                    # Store similarity percentage and result in Redis
                    redis_client.rpush(f"{reference_dir}_similarity_percentages", similarity_percentage)
                    redis_client.rpush(f"{reference_dir}_camera_angle_changes", int(camera_angle_changed))
                    
                    if camera_angle_changed:
                        changed_count += 1
                    else:
                        changed_count = 0

                    if changed_count >= 36:
                        print("Alert: Camera angle changed for 36 consecutive frames.")
                        changed_count = 0

                except Exception as e:
                    print(f"Error processing frame: {e}")
            else:
                print(f"No matching reference frame found for time {current_time_str}")

        time.sleep(60)

# Function to plot the similarity graph
def plot_similarity_graph(timestamps, similarity_scores, reference_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, similarity_scores, marker='o', linestyle='-')
    plt.xlabel('Timestamp (HHMM)')
    plt.ylabel('Similarity Percentage')
    plt.title(f'Similarity Percentage Over Time ({reference_dir})')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to retrieve and plot data from Redis
def plot_data_from_redis(reference_dir):
    timestamps = redis_client.lrange(f"{reference_dir}_timestamps", 0, -1)
    similarity_scores = redis_client.lrange(f"{reference_dir}_similarity_percentages", 0, -1)
    timestamps = [ts.decode('utf-8') for ts in timestamps]
    similarity_scores = [float(score) for score in similarity_scores]
    plot_similarity_graph(timestamps, similarity_scores, reference_dir)

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Retrieving data and generating plot...')
    for i in range(1, len(sys.argv), 2):
        reference_dir = sys.argv[i + 1]
        plot_data_from_redis(reference_dir)
    sys.exit(0)

# Function to process each captured frame for analysis
def process_frame(frame, current_time, reference_frame, similarity_threshold, changed_images_dir, reference_dir):
    # Detect objects and create a mask for the current frame
    detections, img_shape = detect_objects(frame)
    frame_mask = create_mask(frame, detections, img_shape)

    # Calculate similarity with the reference frame using the mask
    similarity_percentage, diff = calculate_similarity(reference_frame, frame, frame_mask)
    
    # Get the current time and date
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    camera_angle_changed = similarity_percentage < similarity_threshold
    status = "possibly changed" if camera_angle_changed else "No change"
    
    print(f"{current_time_str}: Similarity {similarity_percentage:.2f}% - Camera angle {status}.")
    
    # Debugging: Print to verify data is being pushed to Redis
    print(f"Pushing data to Redis for {reference_dir}")
    
    # Push data to Redis
    redis_client.rpush(f"{reference_dir}_similarity_percentages", similarity_percentage)
    redis_client.rpush(f"{reference_dir}_camera_angle_changes", int(camera_angle_changed))
    redis_client.rpush(f"{reference_dir}_timestamps", current_time_str)

    if camera_angle_changed:
        # Concatenate the original and changed frames horizontally
        combined_image = np.hstack((reference_frame, frame))
        
        # Create a save path for the combined image
        save_path = os.path.join(changed_images_dir, f"changed_frame_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
        
        # Save the combined image
        cv2.imwrite(save_path, combined_image)
        print(f"Saved changed frame to {save_path}")
    
    return similarity_percentage, camera_angle_changed

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Global parameters
similarity_threshold = 50  # Similarity threshold in percentage

# Main function to handle inputs and start capturing
def main():
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("Usage: python video-camera-10.py <rtsp_url1> <reference_directory1> [<rtsp_url2> <reference_directory2> ...]")
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
