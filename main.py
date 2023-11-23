import tensorflow as tfimport tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image

# Load the pretrained model
model = tf.keras.applications.VGG16(weights='imagenet')
# Create a Tkinter window
window = tk.Tk()
window.title("Flower Classification App")

# Create a function to handle image drag and drop
def handle_drop(event):
    # Get the file path of the dragged image
    file_path = event.data
    
    # Perform image classification
    # Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize the image to match the model's input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    return image

# Function to perform image classification
def classify_image(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Perform prediction using the loaded model
    predictions = model.predict(image)
    
    # Get the top predicted class
    predicted_class = tf.keras.applications.vgg16.decode_predictions(predictions, top=1)[0][0][1]
    
    return predicted_class

# Handle the image drop or selection
def handle_drop(event):
    file_path = event.data
    predicted_class = classify_image(file_path)
    messagebox.showinfo("Flower Classification", f"The predicted class is: {predicted_class}")

# Perform image classification when opening a file dialog
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        predicted_class = classify_image(file_path)
        messagebox.showinfo("Flower Classification", f"The predicted class is: {predicted_class}")



# Create a function to open a file dialog for image selection
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Perform image classification
        # ...

# Create a drop area and button in the GUI
drop_area = tk.Label(window, text="Drag and drop an image here", width=50, height=10)
drop_area.bind("<Drop>", handle_drop)
drop_area.pack()

open_button = tk.Button(window, text="Open Image", command=open_file_dialog)
open_button.pack()

# Start the GUI event loop
window.mainloop()


