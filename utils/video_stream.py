import cv2


class Camera:

    @staticmethod
    def get_available_cameras():
        """Get list of available camera indices"""
        camera_indices = []
        for i in range(5):  # Check first 5 possible camera indices
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                camera_indices.append(str(i))
                cap.release()
        return camera_indices if camera_indices else ["0"]
    
    @staticmethod
    def load_camera(camera_index):
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise Exception(f"Could not open camera {camera_index}")
        
        return camera