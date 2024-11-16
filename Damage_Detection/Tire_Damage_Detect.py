import torch
import cv2
from yolov5.models.common import DetectMultiBackend  # Import YOLOv5 model loader
from yolov5.utils.general import non_max_suppression  # For handling detections
from yolov5.utils.augmentations import letterbox
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Initialize the YOLOv5 model
model = DetectMultiBackend('C:/CarTire/yolov5/runs/train/exp/weights/best.pt', device='cpu')  # or 'cuda' if GPU is available
model.eval()  # Set to evaluation mode

# Image preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_tire_quality(frame):
    """
    Detects tire defects in the given frame and returns 'Good' or 'Bad' based on detection results.
    Draws bounding boxes if defects are detected.
    :param frame: Input frame from webcam.
    :return: 'Good' if no defects detected, 'Bad' if defects are detected.
    """
    img = letterbox(frame, 640, stride=32, auto=True)[0]  # Preprocess the frame for YOLOv5
    img = img.transpose((2, 0, 1))  # HWC to CHW format
    img = np.ascontiguousarray(img)  # Speed up numpy operations
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # Use float16 if possible
    img /= 255.0  # Normalize image

    if len(img.shape) == 3:
        img = img[None]  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        pred = model(img)  # Run inference
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)  # Apply NMS

    # Initialize a variable to store bounding boxes and detection status
    defect_detected = False
    boxes = []

    # Process detections only if there are valid results
    if pred is not None and len(pred) > 0 and isinstance(pred[0], torch.Tensor):
        for det in pred[0]:  # Iterate over detections for the first image (batch size = 1)
            if det is not None and len(det) >= 6:  # Ensure the detection has enough elements
                x1, y1, x2, y2, conf, cls = det[:6]  # Unpack detection
                boxes.append((int(x1), int(y1), int(x2), int(y2)))  # Store boxes as integer values
                defect_detected = True  # Mark defect detected

    # If defect is detected, return "Bad" and the bounding boxes
    if defect_detected:
        return "Bad", boxes
    return "Good", []  # Otherwise, return "Good" and no boxes

# Start webcam feed and perform real-time tire quality detection
cap = cv2.VideoCapture(1)  # Open the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect tire quality in the current frame
    result, boxes = detect_tire_quality(frame)

    # Display the result on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (30, 50)
    font_scale = 1
    font_color = (0, 255, 0) if result == "Good" else (0, 0, 255)
    thickness = 2
    line_type = cv2.LINE_AA

    # Add text to frame
    cv2.putText(frame, f"Tire Quality: {result}", position, font, font_scale, font_color, thickness, line_type)

    # Draw bounding boxes if defects are detected
    if result == "Bad":
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red box for defects

    # Display the frame
    cv2.imshow("Tire Quality Detection", frame)

    # Press 'q' to exit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()