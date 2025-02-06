# Object-Tracking-App

Assess the candidate&#39;s expertise in developing and deploying advanced
computer vision solutions on edge devices, focusing on real-time performance, model
optimization, and user interface design.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/shuvo881/Object-Tracking-App.git
cd Object-Tracking-App
```

3. Set up environment variables:

```bash
# Create a vertual enviroment
## for Mac and Linux
python -m venv venv

# Active the verual enviroment
## For Mac and Linux
source venv/bin/active
```


```bash
# Install Requirements
pip install -r requirements.txt
```

### Basic Usage

```bash
# run the main file:
python main.py
```


# File Structure

``` bash
Object-Tracking-App/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── Experiment/
│   │   ├── 000000000030.jpg
│   │
│   ├── annotated/
|
│   ├── dataset/
|
│   ├── raw/
│   │   ├── images
│
├── gui/
│   └── main.py
|
├── model_train/
│   └── train.py
|
├── models/
│   └── yolo11n.pt
|
├── utils/
│   └── stats.py
│   └── tracking.py
│   └── video_stream.py
│   └── visualization.py
|
|
├── main.py
├── train_config.yaml

```

Main Project Files:
* README.md: Likely contains project documentation.
* requirements.txt: Contains Python dependencies for the project.
* main.py: Main script for the project, found in the root directory.

Data Files:
* Experiment: Contains experements images
* annotated: Contains annptated images with labels.
* dataset: Contains Train and Val Data
* raw: Contains raw images

GUI File:
* main.py: Contain GUI codes and functionalites

Model Train File:
* train.py: Contain Model Training code part

Models File:
* yolo11n.pt: This is base mode, we use it for training

Utils Files:
* Contain camera chaking, visualization, video_stream and tracking Code



