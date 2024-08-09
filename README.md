# 3.27-detection-of-traffic-in-front-of-a-stop-sign

![8664710](https://github.com/user-attachments/assets/bced2236-c06a-4fed-80cd-a4e66d725258)


# 3.27 'No Stopping' Traffic Sign and Vehicle Detection
### Subtitle: A FastAI-Based Application for Real-Time Detection of 3.27 Traffic Signs and Vehicles with WebSocket Integration

This repository contains an application built using the FastAI library for detecting the 3.27 'No Stopping' traffic sign and various vehicles (cars, buses, trucks). The system monitors video feeds in real-time, detects the presence of the 3.27 traffic sign, and identifies if a vehicle is stopped in front of the sign for more than 10 seconds. If this condition is met, an event is triggered and sent via a WebSocket hook.

### Repository Contents
* app.ipynb: This Jupyter Notebook contains the full implementation of the detection models and the logic for real-time video analysis.
* stop_sign_traffic: A dataset folder containing images of 3.27 'No Stopping' traffic signs used for training the stop sign detection model.
* transport: A dataset folder containing images of various vehicles (cars, buses, trucks) used for training the vehicle detection model.

### Features
* Real-Time Detection: Captures video frames in real-time, detecting 3.27 traffic signs and identifying vehicles in the scene.
* Event Trigger: If a vehicle is detected in front of a 3.27 traffic sign for more than 10 seconds, an event is generated and sent via a WebSocket to a specified server.
* FastAI Models: Leverages FastAI's deep learning capabilities for training and fine-tuning the detection models, ensuring high accuracy.

# Usage
### Prerequisites
Make sure you have the following installed:
```python
pip install fastai opencv-python torch websockets
```
### Running the Application
1. Clone the Repository:
```python
git clone https://github.com/your-repository-url.git
cd your-repository-directory
```
2. Run the Application:
Open the app.ipynb notebook in Jupyter and run all cells to start the application. Alternatively, convert the notebook to a Python script and run it directly:
```python
jupyter nbconvert --to script app.ipynb
python app.py
```
3. WebSocket Integration:
Ensure that the WebSocket server address is correctly configured in the send_event() function within the code.

# Code Overview
1. Stop Sign Detection Model
This model is trained to detect the 3.27 'No Stopping' traffic sign using FastAI.
```python
from fastai.vision.all import *

# Path to your dataset
path = Path('path_to_stop_sign_dataset')

# Define the DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2),
    item_tfms=Resize(224),
    batch_tfms=aug_transforms()
)

# Create DataLoaders
dls = dblock.dataloaders(path, bs=16)

# Create the Learner
learn = cnn_learner(dls, resnet34, metrics=accuracy)

# Fine-tune the model
learn.fine_tune(4)

# Export the model
learn.export('stop_sign_model.pkl')
```
2. Vehicle Detection Model
This model is trained to detect cars, buses, and trucks.
```python
from fastai.vision.all import *

# Path to your dataset
path = Path('path_to_transport_dataset')

def get_y(fname):
    if 'car' in fname.name:
        return 'car'
    elif 'bus' in fname.name:
        return 'bus'
    elif 'truck' in fname.name:
        return 'truck'
    else:
        return 'unknown'

# Define the DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=get_y,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(128),
    batch_tfms=aug_transforms()
)

# Create DataLoaders
dls = dblock.dataloaders(path, bs=16)

# Create the Learner
learn = cnn_learner(dls, resnet34, metrics=accuracy)

# Fine-tune the model
learn.fine_tune(2)

# Export the model
learn.export('vehicle-model.pkl')
```
3. Real-Time Detection and WebSocket Integration
The following code integrates the stop sign and vehicle detection models with real-time video analysis and sends an event if the conditions are met.

```python
import cv2
import torch
from fastai.vision.all import *
import asyncio
import websockets
from pathlib import Path

# Load models
stop_sign_model = load_learner('stop_sign_model.pkl')
vehicle_model = load_learner('vehicle-model.pkl')

# WebSocket server connection
async def send_event():
    uri = "ws://your-websocket-server-address"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Vehicle detected for more than 10 seconds")

# Open video stream
cap = cv2.VideoCapture(0)

stop_sign_detected = False
vehicle_detected_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to PIL format
    img = PILImage.create(frame)

    # Detect stop sign
    stop_sign_pred = stop_sign_model.predict(img)[0]
    if stop_sign_pred == 'stop_sign':
        stop_sign_detected = True
    else:
        stop_sign_detected = False
        vehicle_detected_start_time = None

    if stop_sign_detected:
        # Detect vehicle
        vehicle_pred = vehicle_model.predict(img)[0]
        if vehicle_pred in ['car', 'bus', 'truck']:
            if vehicle_detected_start_time is None:
                vehicle_detected_start_time = cv2.getTickCount()
            else:
                elapsed_time = (cv2.getTickCount() - vehicle_detected_start_time) / cv2.getTickFrequency()
                if elapsed_time > 10:
                    asyncio.run(send_event())
        else:
            vehicle_detected_start_time = None

    # Show output
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
### Model Training and Testing
* Stop Sign Model: Trained using the images from the stop_sign_traffic dataset.
* Vehicle Model: Trained using the transport dataset with categories such as car, bus, and truck.

Both models have been fine-tuned using a ResNet34 architecture for optimal performance in real-time detection scenarios.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions or improvements.

# Acknowledgments
Thanks to the FastAI and OpenCV communities for the tools and resources that made this project possible.
