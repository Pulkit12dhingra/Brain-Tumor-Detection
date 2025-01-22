import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

# Load the pre-trained model
model = keras.models.load_model('/Users/eli/Desktop/brain_tumor_CNN_classifier/final_model')

def classify_image():
    """
    This function opens a file dialog for the user to select an image file.
    It then processes the image and uses the pre-trained model to classify the type of brain tumor.
    The result is displayed in the result_label widget.
    """
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the image
        img = load_img(file_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add an extra dimension to make it rank 4
        datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=keras.applications.vgg16.preprocess_input)
        generator = datagen.flow(x=img)
        
        # Predict the class of the image using the model
        pred = model.predict(generator)
        result = np.argmax(pred, axis=1)
        
        # Display the result in the result_label widget
        if result == 0:
            result_label.config(text='This tumor appears to be a glioma.')
        elif result == 1:
            result_label.config(text='This tumor appears to be a meningioma.')
        elif result == 2:
            result_label.config(text='There does not appear to be a tumor present.')
        elif result == 3:
            result_label.config(text='This tumor appears to be an other tumor type.')
        else:
            result_label.config(text='This tumor appears to be a pituitary tumor.')

# Create the main tkinter window
root = Tk()
root.title('Brain Tumor Classification')

# Create and place widgets
title_label = Label(root, text='What type of brain tumor is this?', font=('Helvetica', 16))