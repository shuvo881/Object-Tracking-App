from gui.main import create_app
from model_train import train 

def main():
    train.main()
    create_app(model_path='runs/detect/train/weights/best.pt')

if __name__ == "__main__":
    main()