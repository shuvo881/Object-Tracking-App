import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from utils.video_stream import Camera

class ObjectDetectionApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Real-time Object Detection")
        self.app.geometry("1200x800")
        
        self.camera = None
        self.is_running = False
        self.video_thread = None
        self.frame_size = (640, 640)  
        
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
            model = YOLO('models/yolo11n.pt')
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def setup_ui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.app)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video frame
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Black background for the video
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="No Camera Feed",
            width=self.frame_size[0],
            height=self.frame_size[1],
            text_color="red",
        )
        self.video_label.pack(fill="both", expand=True)
        
        # Controls frame
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Camera selection dropdown
        self.camera_var = tk.StringVar(value="0")
        self.camera_dropdown = ctk.CTkOptionMenu(
            self.controls_frame,
            values=Camera.get_available_cameras(),
            variable=self.camera_var
        )
        self.camera_dropdown.pack(side="left", padx=5)
        
        # Start/Stop button
        self.toggle_button = ctk.CTkButton(
            self.controls_frame,
            text="Start",
            command=self.toggle_camera
        )
        self.toggle_button.pack(side="left", padx=5)
        
        # FPS display
        self.fps_label = ctk.CTkLabel(self.controls_frame, text="FPS: 0")
        self.fps_label.pack(side="right", padx=5)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Status: Ready",
            text_color="yellow"
        )
        self.status_label.pack(fill="x", padx=10, pady=5)
    
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
                    
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            
            self.is_running = True
            self.toggle_button.configure(text="Stop")
            self.status_label.configure(text="Status: Running", text_color="green")
            
            self.video_thread = threading.Thread(target=self.update_frame)
            self.video_thread.daemon = True  # Thread will close when main window closes
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
    
    def stop_camera(self):
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.toggle_button.configure(text="Start")
        self.status_label.configure(text="Status: Stopped", text_color="yellow")
    
    def update_frame(self):
        last_time = time.time()
        fps_update_interval = 0.1  # Update FPS every 0.1 seconds
        fps_counter = 0
        
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    raise Exception("Failed to grab frame")
                
                # Process frame
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection if model is available
                if self.model is not None:
                    results = self.model(frame)
                    frame = self.draw_detections(frame, results)
                
                # Update UI
                self.update_ui(frame)
                
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - last_time > fps_update_interval:
                    fps = fps_counter / (current_time - last_time)
                    self.fps_label.configure(text=f"FPS: {fps:.1f}")
                    fps_counter = 0
                    last_time = current_time
                
            except Exception as e:
                print(f"Error in update_frame: {e}")
                self.stop_camera()
                self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")
                break
    
    def draw_detections(self, frame, results):
        """Draw detection results on frame"""
        if results and len(results) > 0:
            # Get the first result (we only processed one image)
            result = results[0]
            
            # Draw boxes
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                
                # Add label
                label = f"{result.names[cls]}: {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return frame
    
    def update_ui(self, frame):
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def on_closing(self):
        self.stop_camera()
        self.app.quit()
    
    def run(self):
        self.app.mainloop()

def create_app():
    app = ObjectDetectionApp()
    app.run()