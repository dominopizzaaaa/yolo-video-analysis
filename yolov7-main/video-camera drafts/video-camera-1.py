import cv2
import numpy as np

# Open the video file
video_path = r'C:\Users\Enpag\OneDrive\Desktop\yolo\yolov7-main\inference\videos\cctv.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize variables
total_frames = 0
changed_frames = 0
threshold_percentage = 5  # Adjust this threshold as needed
buffer_percentage = 2  # Adjust this buffer as needed

# Process the first frame to initialize the previous frame
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Unable to read video file")

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create a mask for drawing purposes
mask = np.zeros_like(prev_frame)

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (Lucas-Kanade method)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to detect significant motion
    threshold_magnitude = 1
    motion_mask = magnitude > threshold_magnitude

    # Count the number of pixels with significant motion
    num_pixels_changed = np.count_nonzero(motion_mask)

    # Calculate the percentage of pixels changed relative to the total number of pixels
    frame_height, frame_width = frame.shape[:2]
    total_pixels = frame_height * frame_width
    percentage_changed = (num_pixels_changed / total_pixels) * 100

    # Check if the percentage of changed pixels exceeds the threshold
    if percentage_changed > (threshold_percentage + buffer_percentage):
        changed_frames += 1
        print(f"Frame {total_frames}: Camera angle has changed.")
        # Draw motion vectors on the mask
        for y in range(0, frame_height, 10):
            for x in range(0, frame_width, 10):
                if motion_mask[y, x]:
                    cv2.circle(mask, (x, y), 2, (0, 255, 0), -1)
                    cv2.line(mask, (x, y), (int(x + flow[y, x, 0]), int(y + flow[y, x, 1])), (0, 255, 0), 1)
    else:
        print(f"Frame {total_frames}: Camera angle has not changed.")

    # Display the mask with motion vectors
    cv2.imshow('Motion Vectors', cv2.add(frame, mask))

    # Update the previous frame for the next iteration
    prev_gray = gray

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate the percentage of frames with significant changes
percentage_changed = (changed_frames / total_frames) * 100
print(f"Total frames: {total_frames}")
print(f"Frames with significant camera angle change: {changed_frames}")
print(f"Percentage of frames with camera angle change: {percentage_changed:.2f}%")

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
