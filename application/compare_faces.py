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

import re_train
import add_database

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
            self.name = customtkinter.CTkButton(window, text=self.name, corner_radius=0, border_width=3, fg_color="#000000", text_font=("Helvetica",60), command = self.command, hover_color="#16044a")
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
        collect_passport()

        ########## MAKING PREDICTION ##########
        detection_threshold = 0.6
        verification_threshold = 0.55
        # changes images to appropriate format 
        IMG_PATH = os.path.join("application_data", "input_image")
        VER_PATH = os.path.join("application_data", "verification_images")
        results = []
        for image in os.listdir(VER_PATH):
            input_img = preprocess(os.path.join(IMG_PATH, "input_img.jpg"))
            validation_img = preprocess(os.path.join(VER_PATH, image))
            # makes prediction
            result = siamese_model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        #  comparing results to our required thresholds
        detection = np.sum(np.array(results) > detection_threshold)
        print(detection)
        verification = detection / len(os.listdir(VER_PATH))    
        verified = verification > verification_threshold

        
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


        # adding thew passport img to window
        passport_img = Image.open(passport_image_cap)
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
    
    re_train_button = buttons("Re-train", 4, 1, add_database.add_database)
    re_train_button.button_create()

    # takes screen size
    width, height = pyautogui.size()
    # converts the sizes to fit the label ratio
    width_win_scale = int(width/1.7075)
    height_win_scale = int(height/0.96)
    

    # passport image preview
    passport_button = customtkinter.CTkButton(window, text="Passport Image", text_font = ("Helvetica", 20), fg_color="#000000", hover=False, border_width=2, corner_radius=0, height = 300)
    passport_button.grid(row=1, column=1, sticky=tk.NSEW)

    # creates a border around the image
    passport_frame = customtkinter.CTkFrame(window, highlightbackground="black", highlightthickness=10)
    passport_frame.grid(row=2, column=1, sticky=tk.NSEW)
    # creates a label for the image to be put in
    passport_label = customtkinter.CTkLabel(window, text="")
    passport_label.grid(row=2, column=1, sticky=tk.NSEW, padx = 2, pady=4)

    
    

    # captured image preview
    input_button = customtkinter.CTkButton(window, text="Captured face", text_font = ("Helvetica", 20), fg_color="#000000", hover=False, border_width=2, corner_radius=0, height = 300)
    input_button.grid(row=1, column=2, sticky=tk.NSEW)

    # creates a border around the image
    input_frame = customtkinter.CTkFrame(window, highlightbackground="black", highlightthickness=10)
    input_frame.grid(row=2, column=2, sticky=tk.NSEW)
    # creates a label for the image to be put in
    input_label = customtkinter.CTkLabel(window, text="") 
    input_label.grid(row=2, column=2, sticky=tk.NSEW, padx=2, pady=4)


compare_faces()
