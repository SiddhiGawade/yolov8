from ultralytics import YOLO
import cv2
import numpy as np

def main():
    """
    Alternative YOLO webcam detector with simplified approach.
    Uses the built-in plot() method for easy visualization.
    """
    
    print("Loading your custom YOLO model...")
    # Load your trained model
    model = YOLO("yolov8l-seg.pt")   # Using your custom trained model
    print("Model loaded successfully!")
    
    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
    
    # Set webcam resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        frame_count += 1
        
        # Run YOLO prediction on frame with confidence threshold
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Use YOLO's built-in plot method for easy visualization
        annotated_frame = results[0].plot()
        
        # Add frame counter to the display
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show result in a window
        cv2.imshow("YOLOv8 Custom Model - Live Detection", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"detected_frame_{frame_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Frame saved as {filename}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()