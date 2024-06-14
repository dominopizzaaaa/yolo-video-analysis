import cv2
import os
import sys
import time
import redis
import threading
from datetime import datetime, timedelta

# Set up Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")

def cache_image_in_redis(image, key):
    _, buffer = cv2.imencode('.jpg', image)
    redis_client.set(key, buffer.tobytes())

def capture_frame(rtsp_url, folder_name):
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=24)
    
    while datetime.now() < end_time:
        with SuppressStream(sys.stderr):
            cap = cv2.VideoCapture(rtsp_url)
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

        current_time = datetime.now().strftime("%H%M")
        filename = f"frame_{current_time}.jpg"
        filepath = os.path.join(folder_name, filename)
        
        cv2.imwrite(filepath, frame)
        cache_image_in_redis(frame, filename)
        print(f"Captured and saved {filepath} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with filename {filename}")
        
        cap.release()
        time.sleep(600)  # Wait for 10 minutes before capturing the next frame

    print("24 hours have passed. Stopping the program.")

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

def main():
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("Usage: python retrieve_pictures.py <rtsp_url1> <folder_name1> [<rtsp_url2> <folder_name2> ...]")
        return

    threads = []
    for i in range(1, len(sys.argv), 2):
        rtsp_url = sys.argv[i]
        folder_name = sys.argv[i + 1]
        
        create_folder_if_not_exists(folder_name)
        
        thread = threading.Thread(target=capture_frame, args=(rtsp_url, folder_name))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
