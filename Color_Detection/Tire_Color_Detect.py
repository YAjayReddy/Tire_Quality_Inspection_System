import numpy as np
import cv2
import time
from flask import Flask, render_template, send_file

app = Flask(__name__)

# Define paths for snapshots
snapshot_filename = "snapshot.png"

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Color ranges and detection settings
KNOWN_WIDTH = 24.0
FOCAL_LENGTH = 615.0
black_lower = np.array([0, 0, 0], np.uint8)
black_upper = np.array([180, 255, 50], np.uint8)
gray_lower = np.array([0, 0, 50], np.uint8)
gray_upper = np.array([180, 50, 200], np.uint8)
roi_top_left_x, roi_top_left_y = 100, 100
roi_width, roi_height = 400, 300
capture_interval = 5 # seconds
last_capture_time = 0

# Start capturing frames in a background loop
def capture_frames():
    global last_capture_time
    while True:
        ret, imageFrame = webcam.read()
        if not ret:
            break

        # Define and crop to the region of interest (ROI)
        roi = imageFrame[roi_top_left_y:roi_top_left_y + roi_height,
                         roi_top_left_x:roi_top_left_x + roi_width]
        
        # Convert the ROI frame to HSV color space
        hsvFrame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create masks for black and gray colors
        black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)
        gray_mask = cv2.inRange(hsvFrame, gray_lower, gray_upper)

        # Calculate areas to check color dominance
        black_area = cv2.countNonZero(black_mask)
        gray_area = cv2.countNonZero(gray_mask)
        total_area = roi_width * roi_height

        # Determine dominant color
        majority_threshold = 0.3 * total_area
        if black_area > majority_threshold:
            color_text = "Black"
            frame_color = (0, 255, 0)
        elif gray_area > majority_threshold:
            color_text = "Gray"
            frame_color = (0, 255, 0)
        else:
            color_text = "Unknown Color"
            frame_color = (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(imageFrame, (roi_top_left_x, roi_top_left_y), 
                      (roi_top_left_x + roi_width, roi_top_left_y + roi_height), 
                      frame_color, 2)
        cv2.putText(imageFrame, color_text, 
                    (roi_top_left_x + 10, roi_top_left_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, frame_color, 2)
        
        # Capture image if interval has passed
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            cv2.imwrite(snapshot_filename, imageFrame)
            last_capture_time = current_time

        # Wait briefly
        cv2.waitKey(10)

# Route for the main web page
@app.route('/')
def index():
    return render_template("index.html")

# Route to serve the latest image
@app.route('/image')
def image():
    return send_file(snapshot_filename, mimetype='image/png')

# Start capturing frames in background thread
import threading
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
