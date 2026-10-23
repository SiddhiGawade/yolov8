import cv2
import numpy as np
from ultralytics import YOLO

class YOLOWebcamDetector:
    """
    Simple YOLO object detection using laptop webcam.
    Detects objects and draws bounding boxes in real-time.
    """
    
    def __init__(self, model_path="best.pt"):
        """
        Initialize YOLO model and webcam.
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        print("Loading custom YOLO model...")
        self.model = YOLO(model_path)
        print("Custom YOLO model loaded successfully!")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam")
        
        # Set webcam properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam initialized successfully!")
        print("Press 'q' to quit")
    
    def draw_bounding_boxes(self, frame, results):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: The video frame
            results: YOLO detection results
            
        Returns:
            frame: Frame with bounding boxes drawn
        """
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def run(self):
        """
        Main loop for webcam object detection.
        """
        print("Starting object detection...")
        print("Point your webcam at objects to detect them!")
        
        while True:
            # Read frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Draw bounding boxes
            frame_with_boxes = self.draw_bounding_boxes(frame, results)
            
            # Display the frame
            cv2.imshow('YOLO Object Detection', frame_with_boxes)
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

def main():
    """
    Main function to run the YOLO webcam detector.
    """
    try:
        detector = YOLOWebcamDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the YOLO model file exists.")

if __name__ == "__main__":
    main()
