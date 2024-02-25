import cv2
import json
import numpy as np
from keras.models import load_model

# Load the trained facial recognition model
model = load_model('facial_recognition_model.keras')

# Load and prepare the label map
with open('label_map.json', 'r') as label_file:
    label_map = json.load(label_file)
label_map = {int(k): v for k, v in label_map.items()}

# Load the pre-trained model from OpenCV for face detection
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

def open_gate():
    """Simulate opening a gate."""
    print("Gate opened!")

def preprocess_frame(frame):
    """Preprocess the frame for facial recognition model."""
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model's expected input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension for prediction

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            preprocessed_face = preprocess_frame(face)
            predictions = model.predict(preprocessed_face)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            predicted_class_name = label_map.get(predicted_class, "Unknown")
            if confidence > 0.50 and predicted_class_name != "Unknown":
                open_gate()
                text_label = predicted_class_name
            else:
                text_label = "Unknown"

            # Display the label and bounding box
            cv2.putText(frame, text_label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
