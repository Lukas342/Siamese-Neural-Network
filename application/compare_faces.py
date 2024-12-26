import cv2
import os
import uuid

from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall 

import tkinter as tk
import customtkinter
from PIL import ImageTk, Image

import pyautogui
import numpy as np


def compare_faces():
    global cap
    ########## SETUP ##########
    #customtkinter.set_default_color_theme("dark-blue")
    customtkinter.set_appearance_mode("dark")
    window = customtkinter.CTk()
    window.attributes("-fullscreen", True)
    window.geometry("500x500")

    window.rowconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)
    window.rowconfigure(2, weight=1)
    window.columnconfigure(0, weight=1)

    # class for creating the buttons
    class buttons:
        def __init__(self, name, row, column, command):
            self.name = name
            self.row = row
            self.column = column
            self.command = command

        def button_create(self):
            self.name = customtkinter.CTkButton(window, text=self.name, corner_radius=0, border_width=3, fg_color="#000000", text_font=("Helvetica",60), command=self.command, hover_color="#16044a")
            self.name.grid(row=self.row, column=self.column, sticky=tk.NSEW, columnspan=3)


    # loads our model
    class L1Dist(Layer):
        def __init__(self, **kwargs):
            super().__init__()

        def call(self, input_embedding, validation_embedding):
            return tf.math.abs(input_embedding - validation_embedding)

    siamese_model = tf.keras.models.load_model("siamese_model.h5", custom_objects={"L1Dist":L1Dist, "BinaryCrossentropy":tf.losses.BinaryCrossentropy})


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
        collect_live()

        ########## MAKING PREDICTION ##########
        detection_threshold = 0.7
        verification_threshold = 0.8
        # changes images to appropriate format 
        IMG_PATH = os.path.abspath("application\\application_data\input_image")
        VER_PATH = os.path.abspath("application\\application_data\\verification_images")

        results = []
        for image in os.listdir(VER_PATH):
            input_img = preprocess(os.path.join(IMG_PATH, "input_img.jpg"))
            validation_img = preprocess(os.path.join(VER_PATH, image))
            # makes prediction
            result = siamese_model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        # comparing results to our required thresholds
        detection = np.sum(np.array(results) > detection_threshold)
        print(np.array(results))
        verification = detection / len(os.listdir(VER_PATH))    
        print("Verification: ", verification)
        verified = verification > verification_threshold
        print("Verified: ", verified)

        global match
        global match_colour
        global rectangle_colour
        if verified == True:
            match = "True"
            match_colour = "#00FF00"
            rectangle_colour = (0,255,0)
        else:
            match = "False"
            match_colour = "#FF0000"
            rectangle_colour = (0,0,255)

        match_label.configure(text=match, fg_color=match_colour)

        # adding the passport img to window
        passport_img_path = os.path.abspath("application\\application_data\\verification_images\passport_img.jpg")
        passport_img = Image.open(passport_img_path)

        passport_img = passport_img.resize((300, 300)) 

        passport_img_tk = ImageTk.PhotoImage(image=passport_img)
        passport_label.passport_img_tk = passport_img_tk 
        passport_label.configure(image=passport_img_tk)

        # adds the image to the label
        input_img = Image.open(live_image_cap)
        input_img = input_img.resize((300, 300))
        input_img_tk = ImageTk.PhotoImage(image=input_img)
        input_label.input_img_tk = input_img_tk
        input_label.configure(image=input_img_tk)    


    ########## GUI ##########
    match = "Face recogniton"
    match_colour = "#000000"

    # place the buttons
    match_label = customtkinter.CTkLabel(window, text=match, corner_radius=0, fg_color=match_colour, text_font=("Helvetica",60))
    match_label.grid(row=0, column=1, sticky=tk.NSEW, columnspan=3)

    compare_button = buttons("Compare Face", 3, 1, button_pressed)
    compare_button.button_create()

    # takes screen size
    width, height = pyautogui.size()
    # converts the sizes to fit the label ratio
    width_win_scale = int(width/1.7075)
    height_win_scale = int(height/0.96)

    # passport image preview
    passport_button = customtkinter.CTkButton(window, text="Passport Image", text_font=("Helvetica", 20), fg_color="#000000", hover=False, border_width=2, corner_radius=0, height=300)
    passport_button.grid(row=1, column=1, sticky=tk.NSEW)

    # creates a border around the image
    passport_frame = customtkinter.CTkFrame(window, highlightbackground="black", highlightthickness=10)
    passport_frame.grid(row=2, column=1, sticky=tk.NSEW)
    # creates a label for the image to be put in
    passport_label = customtkinter.CTkLabel(window, text="")
    passport_label.grid(row=2, column=1, sticky=tk.NSEW, padx=2, pady=4)

    # captured image preview
    input_button = customtkinter.CTkButton(window, text="Captured face", text_font=("Helvetica", 20), fg_color="#000000", hover=False, border_width=2, corner_radius=0, height=300)
    input_button.grid(row=1, column=2, sticky=tk.NSEW)

    # creates a border around the image
    input_frame = customtkinter.CTkFrame(window, highlightbackground="black", highlightthickness=10)
    input_frame.grid(row=2, column=2, sticky=tk.NSEW)
    # creates a label for the image to be put in
    input_label = customtkinter.CTkLabel(window, text="") 
    input_label.grid(row=2, column=2, sticky=tk.NSEW, padx=2, pady=4)

    cap = cv2.VideoCapture(1)
    cam_label = tk.Label(window)
    cam_label.grid(row=0, column=0, sticky=tk.NSEW, rowspan=4)

    global rectangle_colour
    rectangle_colour = (255,0,0)
    def frame_capture():
        global rectangle_colour
        # to capture live feed we essentially need to convert to picture frame by frame
        frame = cap.read()[1]

        # drawing a box around the detected face
        haar_get = os.path.abspath("D:\Github\Siamese-neural-network\data\haarcascade_frontalface_default.xml")
        haar_cascade = cv2.CascadeClassifier(haar_get)
        # converting to greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect the face
        faces = haar_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)

        # get the co-ordinates and draw around them
        for (x,y,w,h) in faces:
            width_cords = x+w
            height_cords = y+h
            cv2.rectangle(frame, (x,y), (width_cords,height_cords), rectangle_colour, 2)

        frame = cv2.resize(frame, (width_win_scale,height_win_scale))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        img_tk = ImageTk.PhotoImage(image=img)

        cam_label.img_tk = img_tk
        cam_label.configure(image=img_tk)

        cam_label.after(10, frame_capture)

    frame_capture()        
    window.mainloop()

########## Image collection ##########
# collect live image
def collect_live():
    global cap
    ANC_PATH = os.path.abspath("application\\application_data\input_image")

    while cap.isOpened():
        ret, frame = cap.read()
        # Cut down frame to 250x250px
        frame = frame[120 : 120 + 250, 200 : 200 + 250, :]

        # Collect anchors
        for image in os.listdir(ANC_PATH):
            image_remove = os.path.join(ANC_PATH, image)
            os.remove(image_remove)
        for i in range(1):
            # Create the unique file path
            imgname = os.path.join(ANC_PATH, "input_img.jpg")
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        break

    # write the image collected
    for image in os.listdir(ANC_PATH):
        global live_image_cap
        live_image_cap = os.path.join(ANC_PATH, image)