# 3.27-detection-of-traffic-in-front-of-a-stop-sign

![8664710](https://github.com/user-attachments/assets/bced2236-c06a-4fed-80cd-a4e66d725258)


# 🛑 Automated Traffic Violation Detection (3.27 Sign Enforcement)

## 📝 Project Overview
This project aims to automate the detection of traffic violations in "No Stopping" (3.27) zones. Using **Computer Vision** and **Deep Learning**, the system monitors a specific area to detect:
1. The presence of a **3.27 (No Stopping)** road sign.
2. The presence of vehicles (**car, bus, truck**) within the sign's jurisdiction.
3. If a vehicle remains stationary for more than **10 seconds**, the system automatically triggers a **WebSocket event** to a central server (simulating a fine or notification system).

## 🛠️ Tools & Technologies
* **Deep Learning Framework:** Fastai (built on PyTorch)
* **Pre-trained Model:** ResNet34 (Transfer Learning)
* **Computer Vision:** OpenCV
* **Real-time Communication:** WebSockets & Asyncio
* **Programming Language:** Python

## 🚀 How it Works
### 1. Dual-Model Architecture
* **Sign Detection Model:** A ResNet34 model trained to identify the 3.27 traffic sign.
* **Vehicle Classification Model:** A separate ResNet34 model trained to categorize vehicles (Car, Bus, Truck).

### 2. Detection Logic
The system uses a state-machine logic:
* If a **3.27 sign** is detected, it activates the vehicle monitor.
* If a **vehicle** is then detected, a timer starts.
* If the vehicle stays for **>10 seconds**, an `async` WebSocket signal is sent.

## 📂 Project Structure
* `stop_sign_model.pkl`: Exported Fastai learner for sign detection.
* `vehicle-model.pkl`: Exported Fastai learner for vehicle classification.
* `traffic_monitor.py`: The main execution script with OpenCV integration.

---
*Created by [Sardor] - Specialized in AI & Computer Vision Solutions*
