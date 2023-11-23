import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Load and preprocess the image data
# Define the paths to the dataset
dataset_dir = 'path/to/dataset'  # Directory where the dataset is stored
image_dir = os.path.join(dataset_dir, 'images')  # Directory containing the images
label_file = os.path.join(dataset_dir, 'labels.txt')  # File containing the labels

# Load the labels
with open(label_file, 'r') as file:
    labels = file.read().splitlines()

# Load the images and labels
images = []
image_labels = []
for i, label in enumerate(labels):
    image_path = os.path.join(image_dir, f'{i+1}.jpg')
    image = load_img(image_path, target_size=(224, 224))  # Adjust the target size as needed
    image = img_to_array(image)
    images.append(image)
    image_labels.append(label)

# Convert the images and labels to numpy arrays
images = np.array(images)
image_labels = np.array(image_labels)





# Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=num_epochs, validation_data=(val_images, val_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Make predictions
predictions = model.predict(new_images)
