# utils/stats.py
class DetectionStats:
    def __init__(self):
        self.class_counts = {}
        self.total_detections = 0
        self.avg_confidence = 0
        self.active_tracks = set()
    
    def update(self, results, track_matches):
        """Update detection statistics"""
        if not results or len(results) == 0:
            return
            
        result = results[0]
        boxes = result.boxes
        
        # Update total detections
        num_detections = len(boxes)
        self.total_detections += num_detections
        
        # Update active tracks
        self.active_tracks.update(track_matches.values())
        
        # Update class counts and average confidence
        total_conf = 0
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = result.names[cls]
            
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
            total_conf += conf
        
        # Update average confidence
        if num_detections > 0:
            self.avg_confidence = (
                self.avg_confidence * (self.total_detections - num_detections) +
                total_conf * 100
            ) / self.total_detections
    
    def get_summary(self):
        """Get formatted statistics summary"""
        stats = {
            "total_detections": self.total_detections,
            "avg_confidence": self.avg_confidence,
            "unique_objects": len(self.active_tracks),
            "class_counts": self.class_counts
        }
        return stats