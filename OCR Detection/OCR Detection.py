import torch
import cv2
from PIL import Image
import pytesseract
import os

# Load the YOLOv5 model (with error handling)
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit(1)

def detect_and_extract_text():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return None
    
    ret, frame = cap.read()

    # Check if frame was captured correctly
    if not ret:
        print("Failed to capture image.")
        cap.release()
        return None

    # Run YOLOv5 on the frame to detect characters
    results = model(frame)
    predictions = results.pandas().xyxy[0]  # Predictions as a DataFrame

    # Loop through each detected character
    extracted_text = ""
    for _, row in predictions.iterrows():
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Crop detected character
        cropped = frame[y_min:y_max, x_min:x_max]

        # Perform OCR on the cropped image
        char_text = pytesseract.image_to_string(Image.fromarray(cropped), config='--psm 10')
        extracted_text += char_text.strip()

        # Draw bounding boxes (optional)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the result with bounding boxes
    cv2.imshow("Character Detection", frame)
    cv2.waitKey(0)  # Wait indefinitely for the user to close the window

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return extracted_text

if __name__ == "__main__":
    text = detect_and_extract_text()
    
    if text:  # Only print and save if text is detected
        print("Extracted Text: ", text)

        # Create the "images" folder if it doesn't exist
        if not os.path.exists("images"):
            os.makedirs("images")

        # Save the frame with bounding boxes
        cv2.imwrite("images/sample_tire_with_bboxes.jpg", frame)

        # Save the extracted text to a text file
        with open("extracted_text.txt", "w") as f:
            f.write(text)
    else:
        print("No text detected.")
