# utils/tracking.py
import numpy as np
import colorsys
import cv2

class ObjectTracker:
    def __init__(self):
        self.track_history = {}  # Track history for each ID
        self.id_colors = {}      # Color mapping for each ID
        self.last_positions = {} # Last known positions
        self.lost_tracks = {}    # Recently lost tracks
        self.max_history = 30
        self.max_lost_frames = 30
        self.iou_threshold = 0.3
        
    def get_color(self, track_id):
        """Generate consistent color for track ID"""
        if track_id not in self.id_colors:
            hue = (track_id * 0.1) % 1.0
            rgb_color = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 1.0))
            self.id_colors[track_id] = rgb_color
        return self.id_colors[track_id]
    
    def calculate_iou(self, box1, box2):
        """Calculate IOU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def match_tracks(self, detections):
        """Match current detections with existing tracks"""
        if not self.last_positions:
            return {i: i for i in range(len(detections))}
            
        matches = {}
        used_tracks = set()
        used_detections = set()
        
        # First, try to match with active tracks
        for det_idx, det_box in enumerate(detections):
            best_iou = self.iou_threshold
            best_track = None
            
            for track_id, last_box in self.last_positions.items():
                if track_id in used_tracks:
                    continue
                    
                iou = self.calculate_iou(det_box, last_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            
            if best_track is not None:
                matches[det_idx] = best_track
                used_tracks.add(best_track)
                used_detections.add(det_idx)
        
        # Try to match with recently lost tracks
        for det_idx in range(len(detections)):
            if det_idx in used_detections:
                continue
                
            best_iou = self.iou_threshold
            best_track = None
            
            for track_id, (box, frames_lost) in list(self.lost_tracks.items()):
                if frames_lost > self.max_lost_frames:
                    del self.lost_tracks[track_id]
                    continue
                    
                iou = self.calculate_iou(detections[det_idx], box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            
            if best_track is not None:
                matches[det_idx] = best_track
                del self.lost_tracks[best_track]
                used_detections.add(det_idx)
        
        # Assign new IDs to unmatched detections
        next_id = max(self.last_positions.keys(), default=-1) + 1
        for det_idx in range(len(detections)):
            if det_idx not in matches:
                matches[det_idx] = next_id
                next_id += 1
        
        return matches
    
    def update(self, detections, boxes, frame=None):
        """Update tracks with new detections"""
        # Match detections with existing tracks
        matches = self.match_tracks([box.xyxy[0].cpu().numpy() for box in boxes])
        
        # Update last positions and track history
        new_positions = {}
        for det_idx, track_id in matches.items():
            box = boxes[det_idx]
            bbox = box.xyxy[0].cpu().numpy()
            new_positions[track_id] = bbox
            
            # Update track history
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            
            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id].pop(0)
        
        # Update lost tracks
        for track_id in list(self.last_positions.keys()):
            if track_id not in new_positions:
                self.lost_tracks[track_id] = (self.last_positions[track_id], 0)
        
        # Increment lost frames counter
        for track_id in list(self.lost_tracks.keys()):
            box, frames_lost = self.lost_tracks[track_id]
            self.lost_tracks[track_id] = (box, frames_lost + 1)
            
            # Remove old lost tracks
            if frames_lost > self.max_lost_frames:
                del self.lost_tracks[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]
        
        self.last_positions = new_positions
        return matches