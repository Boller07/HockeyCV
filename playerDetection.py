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
            frame_ = results[0].plot()
            
            cv2.imshow('Player Detection', frame_)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Usage
detector = PlayerDetector()
detector.detect_players("your_video_path.mp4")
