from ultralytics import YOLO
import cv2

class PlayerDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        
    def detect_players(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret = True

        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.track(frame, persist=True)
            detections = results[0].boxes

            if detections is not None and detections.xyxy is not None:
                for box in detections.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box[:4]

                    # Bottom center of the bounding box
                    center = (int((x1 + x2) / 2), int(y2))

                    # Small horizontal ellipse (feet area), width is ~half the box, height is very small
                    width = int((x2 - x1) * 0.5)
                    height = int((y2 - y1) * 0.1)

                    axes = (width, height)
                    cv2.ellipse(frame, center, axes, angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=2)
            
            cv2.imshow('Player Detection', frame_)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Usage
detector = PlayerDetector()
detector.detect_players("your_video_path.mp4")
