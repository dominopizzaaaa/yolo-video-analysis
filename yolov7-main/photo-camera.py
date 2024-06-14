import cv2
import os
import numpy as np
from scipy.ndimage import laplace

def measure_blur_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = laplace(gray)
    blur_intensity = np.var(laplacian)
    return blur_intensity

def detect_night_conditions(image, brightness_threshold=56):
    # Assuming the top 10% of the image contains the sky region
    sky_region = image[:image.shape[0] // 10, :]
    gray_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_sky)
    brightness_percentage = (mean_brightness / 255) * 100
    return mean_brightness < brightness_threshold, brightness_percentage

def adjust_blur_threshold(brightness_percentage, base_threshold=12550):
    # Adjust the threshold based on brightness
    if brightness_percentage < 40:
        adjusted_threshold = base_threshold + 2000  # Increase threshold for very low brightness
    elif brightness_percentage < 50:
        adjusted_threshold = base_threshold + 1000  # Moderate increase for low brightness
    else:
        adjusted_threshold = base_threshold  # Use base threshold for normal brightness
    return adjusted_threshold

def detect_water_droplets(image, blur_threshold=12550):
    blur_intensity = measure_blur_intensity(image)
    return blur_intensity

def test_detect_water_droplets(directory_path):
    # Get all image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        # Load the test frame
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        # Detect night conditions
        night_conditions, brightness_percentage = detect_night_conditions(frame)
        
        # Adjust blur threshold based on brightness
        adjusted_blur_threshold = adjust_blur_threshold(brightness_percentage)
        
        # Detect water droplets
        blur_intensity = detect_water_droplets(frame, blur_threshold=adjusted_blur_threshold)

        # Print the blur intensity and brightness percentage
        print(f"{image_file}: Blur intensity: {blur_intensity:.2f}, Brightness: {brightness_percentage:.2f}%")

if __name__ == "__main__":
    # Replace with the path to your test directory
    test_directory_path = r"C:\Users\Enpag\OneDrive\Desktop\yolo\yolov7-main\videos02"
    test_detect_water_droplets(test_directory_path)

# 1251, 1302, 1312, 1322, 1332, 1342, 1353, 1403, 1414, 1602, 1612, 1622, 1632, 1642, 1652, 1702, 1712, 1723, 1733, 1743, 1753