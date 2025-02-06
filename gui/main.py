import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
import numpy as np
from datetime import datetime
import os

from utils.video_stream import Camera
from utils.tracking import ObjectTracker
from utils.visualization import Visualizer
from utils.stats import DetectionStats

class ObjectDetectionApp:
    def __init__(self, model_path):
        self.app = ctk.CTk()
        self.app.title("Real-time Object Detection and Tracking")
        self.app.geometry("1200x800")
        self.model_path = model_path
        
        # Initialize core attributes
        self.camera = None
        self.is_running = False
        self.video_thread = None
        self.frame_size = (640, 640)
        self.recording = False
        self.video_writer = None
        
        # Initialize tracking and visualization
        self.tracker = ObjectTracker()
        self.visualizer = Visualizer(self.tracker)
        self.stats = DetectionStats()
        
        # Create output directory
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        try:
            self.model = self.load_model()
            if self.model is None:
                raise Exception("Failed to load model")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
            self.model = None
        
        # Setup UI
        self.setup_ui()
        
        # Bind window close event
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self):
        try:
            model = YOLO(self.model_path)  # or your custom model path
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def setup_ui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.app)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel for controls and stats
        self.left_panel = ctk.CTkFrame(self.main_frame, width=200)
        self.left_panel.pack(side="left", fill="y", padx=5, pady=5)
        
        # Stats frame
        self.stats_frame = ctk.CTkFrame(self.left_panel)
        self.stats_frame.pack(fill="x", padx=5, pady=5)
        
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Detection Statistics",
            font=("Arial", 14, "bold")
        )
        self.stats_label.pack(pady=5)
        
        self.total_detections_label = ctk.CTkLabel(
            self.stats_frame,
            text="Total Detections: 0"
        )
        self.total_detections_label.pack()
        
        self.avg_conf_label = ctk.CTkLabel(
            self.stats_frame,
            text="Avg Confidence: 0%"
        )
        self.avg_conf_label.pack()
        
        self.unique_objects_label = ctk.CTkLabel(
            self.stats_frame,
            text="Unique Objects: 0"
        )
        self.unique_objects_label.pack()
        
        self.class_counts_text = ctk.CTkTextbox(
            self.stats_frame,
            height=100,
            width=180
        )
        self.class_counts_text.pack(pady=5)
        
        # Video frame
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="No Camera Feed",
            width=self.frame_size[0],
            height=self.frame_size[1],
            text_color="red"
        )
        self.video_label.pack(fill="both", expand=True)
        
        # Controls frame
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Camera controls
        self.camera_var = tk.StringVar(value="0")
        self.camera_dropdown = ctk.CTkOptionMenu(
            self.controls_frame,
            values=Camera.get_available_cameras(),
            variable=self.camera_var
        )
        self.camera_dropdown.pack(side="left", padx=5)
        
        self.toggle_button = ctk.CTkButton(
            self.controls_frame,
            text="Start",
            command=self.toggle_camera
        )
        self.toggle_button.pack(side="left", padx=5)
        
        # Recording button
        self.record_button = ctk.CTkButton(
            self.controls_frame,
            text="Start Recording",
            command=self.toggle_recording,
            state="disabled"
        )
        self.record_button.pack(side="left", padx=5)
        
        # FPS display
        self.fps_label = ctk.CTkLabel(
            self.controls_frame,
            text="FPS: 0"
        )
        self.fps_label.pack(side="right", padx=5)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Status: Ready",
            text_color="yellow"
        )
        self.status_label.pack(fill="x", padx=10, pady=5)
    
    def toggle_recording(self):
        if not self.recording and self.is_running:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"detection_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                30.0,
                self.frame_size
            )
            self.recording = True
            self.record_button.configure(text="Stop Recording")
            self.status_label.configure(text="Status: Recording", text_color="red")
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            self.record_button.configure(text="Start Recording")
            if self.is_running:
                self.status_label.configure(text="Status: Running", text_color="green")
    
    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        try:
            camera_index = int(self.camera_var.get())
            self.camera = Camera.load_camera(camera_index)
            
            if self.camera:
                self.video_label.configure(text="")
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
                
                self.is_running = True
                self.toggle_button.configure(text="Stop")
                self.status_label.configure(text="Status: Running", text_color="green")
                self.record_button.configure(state="normal")
                
                self.video_thread = threading.Thread(target=self.update_frame)
                self.video_thread.daemon = True
                self.video_thread.start()
                
        except Exception as e:
            error_msg = f"Error starting camera: {e}"
            print(error_msg)
            messagebox.showerror("Camera Error", error_msg)
            if self.camera is not None:
                self.camera.release()
            self.camera = None
            self.is_running = False
            self.toggle_button.configure(text="Start")
            self.status_label.configure(text="Status: Error", text_color="red")
            self.record_button.configure(state="disabled")
    
    def stop_camera(self):
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.toggle_button.configure(text="Start")
        self.status_label.configure(text="Status: Stopped", text_color="yellow")
        self.record_button.configure(state="disabled")
        if self.recording:
            self.toggle_recording()
    
    def update_statistics(self, stats):
        self.total_detections_label.configure(
            text=f"Total Detections: {stats['total_detections']}"
        )
        self.avg_conf_label.configure(
            text=f"Avg Confidence: {stats['avg_confidence']:.1f}%"
        )
        self.unique_objects_label.configure(
            text=f"Unique Objects: {stats['unique_objects']}"
        )
        
        class_counts_str = "Class Counts:\n"
        for cls, count in sorted(stats['class_counts'].items()):
            class_counts_str += f"{cls}: {count}\n"
        self.class_counts_text.delete("1.0", tk.END)
        self.class_counts_text.insert("1.0", class_counts_str)
    
    def draw_detections(self, frame, results):
        if results and len(results) > 0:
            # Update tracking and draw detections
            track_matches = self.tracker.update(results, results[0].boxes)
            frame = self.visualizer.draw_detections(frame, results, track_matches)
            
            # Update statistics
            self.stats.update(results, track_matches)
            self.update_statistics(self.stats.get_summary())
        
        return frame
    
    def update_frame(self):
        last_time = time.time()
        fps_update_interval = 0.1
        fps_counter = 0
        
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    raise Exception("Failed to grab frame")
                
                # Process frame
                frame = cv2.resize(frame, self.frame_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection with tracking
                if self.model is not None:
                    results = self.model(frame_rgb, verbose=False)
                    frame_rgb = self.draw_detections(frame_rgb, results)
                
                # Record if active
                if self.recording and self.video_writer:
                    self.video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                
                # Update UI
                self.update_ui(frame_rgb)
                
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - last_time > fps_update_interval:
                    fps = fps_counter / (current_time - last_time)
                    self.fps_label.configure(text=f"FPS: {fps:.1f}")
                    fps_counter = 0
                    last_time = current_time
                
                # Small delay to prevent high CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in update_frame: {e}")
                self.stop_camera()
                self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")
                break
    
    def update_ui(self, frame):
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def on_closing(self):
        self.stop_camera()
        if self.video_writer:
            self.video_writer.release()
        self.app.quit()
    
    def run(self):
        
        self.app.mainloop()

def create_app(model_path):
    app = ObjectDetectionApp(model_path)
    app.run()