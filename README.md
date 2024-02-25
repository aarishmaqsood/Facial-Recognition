# Facial Recognition-Based Gate Access Control

This project uses facial recognition to control gate access, automatically opening the gate for recognized individuals and identifying any unknown individuals. It leverages a trained Keras model with MobileNetV2 architecture and OpenCV's DNN module for face detection.

## Project Structure

- `dataset/`: Contains training and testing images organized by individual names.
- `facial_recognition_training.py`: Script to train the facial recognition model.
- `gate_access_control.py`: Main script for gate access control using facial recognition.
- `deploy.prototxt.txt` & `res10_300x300_ssd_iter_140000.caffemodel`: OpenCV DNN model files for face detection.
- `facial_recognition_model.keras`: Trained Keras model for facial recognition.
- `label_map.json`: JSON file mapping class labels to their respective names.
- `requirements.txt`: Contains the list of packages required to run the project.

## Setup

1. **Install Dependencies**: Ensure you have Python 3.10 installed and then install the required Python packages using:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset**: Organize your dataset according to the following structure:

   ```
   Dataset/
   ├── Train/
   │   ├── Person1/
   │   │   ├── Pic1.jpg
   │   │   ├── Pic2.jpg
   │   │   ...
   │   ├── Person2/
   │   ...
   ├── Test/
       ├── Person1/
       ├── Person2/
       ...
   ```

3. **Train the Model**: If you haven't already, train the facial recognition model by running:

   ```bash
   python facial_recognition_training.py
   ```

   Make sure `facial_recognition_model.keras` and `label_map.json` are generated in the project directory.

4. **Running Gate Access Control**: Execute the `gate_access_control.py` script to start the gate access control system. Make sure your camera is connected and properly configured.

   ```bash
   python gate_access_control.py
   ```

## Usage

When `gate_access_control.py` is running, it will use your camera to detect and recognize faces in real-time. Known individuals (those present in the training dataset) will trigger the gate to open, while unknown individuals will be tagged as "Unknown" in the video feed.

For best results, ensure good lighting conditions and that the faces in the dataset are varied and cover different angles.
