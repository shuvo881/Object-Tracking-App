from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("models/yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model with 2 GPUs
    results = model.train(data="train_config.yaml", epochs=120, imgsz=640)