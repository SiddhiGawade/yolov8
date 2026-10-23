from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8l-seg.pt")   # using standard YOLOv8 large segmentation model

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO prediction on frame
    results = model.predict(frame, conf=0.5, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show result in a window
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
