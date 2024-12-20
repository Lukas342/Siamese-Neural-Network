import cv2
import os

from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall 

#import collect_passport
#import collect_live

import tkinter as tk
import customtkinter
from PIL import ImageTk, Image

def compare_faces():
    # sets theame colours
    customtkinter.set_default_color_theme("green")
    customtkinter.set_appearance_mode("dark")
    # creates the window
    window = customtkinter.CTk()
    # sets window to fullscreen
    window.attributes("-fullscreen", True)
    window.geometry("500x500")

    # creates our distance layer
    class L1Dist(Layer):
        def __init__(self, **kwargs):
            super().__init__()
            
        def call(self, input_embedding, validation_embedding):
            return tf.math.abs(input_embedding - validation_embedding)
    # loads the model
    siamese_model = tf.keras.models.load_model(r"C:\Users\lukas\OneDrive - Tunbridge Wells Grammar School for Boys\Greenhalf Lukas\Coursework\Code\Siamese neural network\siamese_model.h5",
                                               custom_objects={"L1Dist":L1Dist, "BinaryCrossentropy":tf.losses.BinaryCrossentropy})
    
    
    # class for creating the buttons
    class buttons:
        def __init__(self, name, row, column):
            self.name = name
            self.row = row
            self.column = column
    
        def button_create(self):
            self.name = customtkinter.CTkButton(window, text=self.name, corner_radius=0, border_width=10, fg_color="#808080", text_font=("Roman",60), command = button_pressed)
            self.name.grid(row=self.row, column=self.column, sticky=tk.NSEW)



    def preprocess(file_path):
        
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    def preprocess_twin(input_img, validation_img, label):
        return(preprocess(input_img), preprocess(validation_img), label)

    def button_pressed():
        # collects the images
        collect_live.collect_live()
        collect_passport.collect_passport()

        # changes images to appropriate format 
        POS_PATH = os.path.join("application_data", "input_image")
        NEG_PATH = os.path.join("application_data", "negative_images")
        ANC_PATH = os.path.join("application_data", "verification_images")

        anchor = tf.data.Dataset.list_files(ANC_PATH+"/*.jpg").take(300)
        positive = tf.data.Dataset.list_files(POS_PATH+"/*.jpg").take(300)
        negative = tf.data.Dataset.list_files(NEG_PATH+"/*.jpg").take(300)


        positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
        negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
        data = positives.concatenate(negatives)


        # Build dataloader pipeline
        data = data.map(preprocess_twin)
        data = data.cache()
        #data = data.shuffle(buffer_size=1024)

        # Training partition
        train_data = data.take(round(len(data)*.7))
        train_data = train_data.batch(16)
        train_data = train_data.prefetch(8)

        # Testing partition
        test_data = data.skip(round(len(data)*.7))
        test_data = test_data.take(round(len(data)*.3))
        test_data = test_data.batch(16)
        test_data = test_data.prefetch(8)

        test_input, test_validation, y_true = test_data.as_numpy_iterator().next()
        # making the prediction
        y_pred = siamese_model.predict([test_input, test_validation])

        print(y_pred)

        results = []
        for prediction in y_pred:
            if prediction < 0.6:
                results.append(1)
            else:
                results.append(0)
        print(results)
        print(y_true)
        # Creating a metric object 
        m = Recall()
        # Calculating the recall value 
        m.update_state(y_true, y_pred)
        # Return Recall Result
        print(m.result().numpy())

        if m.result().numpy() >= 0.875:
            print("True")
        else:
            print("False")
            
            
    # place the buttons
    compare_button = buttons("Compare faces", 1, 1)
    compare_button.button_create()
    

    cap = cv2.VideoCapture(0)
    cap.set(3, 500)
    cap.set(4, 500)
    cam_label = tk.Label(window)

    cam_label.grid(row=0, column=0, sticky=tk.NW)
    
    def frame_capture():
        # to capture live feed we essentially need to convert to picture frame by frame
        frame = cap.read()[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)

        cam_label.img_tk = img_tk
        cam_label.configure(image=img_tk)
        cam_label.after(10, frame_capture)

    frame_capture()    
    
    window.mainloop()
    
compare_faces()