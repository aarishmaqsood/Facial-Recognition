import os
import cv2
import json
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Define image dimensions
img_height, img_width = 224, 224

# Function to preprocess images
def preprocess_image(image):
    """Normalize image array."""
    return image / 255.0

# Directories for training and testing data
train_dir = 'dataset/Train'
test_dir = 'dataset/Test'

# Image data generators for training and validation
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image,
                                   validation_split=0.2)  # 20% of data for validation

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Load and configure the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Final model setup
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=10)

# Prepare the test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Function to load and preprocess a single image
def load_and_preprocess_image(img_path, img_height=224, img_width=224):
    """Load and preprocess image for model prediction."""
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_image(img_array_expanded_dims)

# Predict on a custom image
custom_img_path = 'dataset/Test/Mohamed/7.jpg'
preprocessed_image = load_and_preprocess_image(custom_img_path)
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_class_probability = np.max(predictions)
print(f"Predicted class: {predicted_class} with probability {predicted_class_probability}")

# Mapping class indices to class labels
label_map = (train_generator.class_indices)
label_map = dict((v, k) for k, v in label_map.items())
predicted_class_name = label_map[predicted_class]
print(f"Predicted class name: {predicted_class_name}")

# Save the class labels to a JSON file
with open('label_map.json', 'w') as label_file:
    json.dump(label_map, label_file)

# Save the trained model
model.save('facial_recognition_model.keras')
