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
            
            cv2.imshow('Player Detection', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def get_ice_mask(self, frame):
        """
        Returns a binary mask of likely ice areas by thresholding white shades in HSV.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White = low saturation, high brightness. Range tuned to include less pure white too.
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 60, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Use morphological ops to clean up small holes and noise in the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_detections(self, frame):
        """
        Runs YOLOv8 tracking on the frame and returns bounding boxes.
        """
        results = self.model.track(frame, persist=True)
        return results[0].boxes if results else None

    def draw_player_ellipses(self, frame, detections, mask):
        """
        Draws an ellipse at the feet of each detected player if they are over the ice.
        """
        if detections is None or detections.xyxy is None:
            return frame
        
        height, width = mask.shape[:2]
        for box in detections.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            center = (int((x1 + x2) / 2), int(y2))

            # Find coordinates of box that are inside frame
            cx = max(0, min(center[0], width - 1))
            cy = max(0, min(center[1], height - 1))

            # Only draw if bottom of box is on ice
            if mask[cy, cx] == 255:
                width_ellipse = int((x2 - x1) * 0.5)
                height_ellipse = int((y2 - y1) * 0.1)
                axes = (width_ellipse, height_ellipse)
                cv2.ellipse(frame, center, axes, angle=0, startAngle=0, endAngle=360,
                            color=(0, 255, 0), thickness=2)
        return frame

# Usage
detector = PlayerDetector()
detector.detect_players("your_video_path.mp4")
