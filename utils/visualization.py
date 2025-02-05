# utils/visualization.py
import cv2
import numpy as np

class Visualizer:
    def __init__(self, tracker):
        self.tracker = tracker
    
    def draw_detection(self, frame, box, track_id, class_name, confidence):
        """Draw a single detection with its track ID and trail"""
        x1, y1, x2, y2 = map(int, box)
        color = self.tracker.get_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        # Prepare label
        label = f"{class_name} ID:{track_id} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - 25),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Draw trail if available
        if track_id in self.tracker.track_history:
            points = np.array(self.tracker.track_history[track_id], np.int32)
            cv2.polylines(
                frame,
                [points],
                False,
                color,
                2,
                cv2.LINE_AA
            )
        
        return frame
    
    def draw_detections(self, frame, results, track_matches):
        """Draw all detections and their trails"""
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get tracking ID
                track_id = track_matches.get(i)
                if track_id is None:
                    continue
                
                # Get detection info
                bbox = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = result.names[cls]
                
                # Draw detection
                frame = self.draw_detection(
                    frame,
                    bbox,
                    track_id,
                    class_name,
                    conf
                )
        
        return frame