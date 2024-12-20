import cv2
import os
import uuid

from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall 

import collect_passport
import collect_live

import tkinter as tk
import customtkinter
from PIL import ImageTk, Image

import pyautogui


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
            self.name = customtkinter.CTkButton(window, text=self.name, corner_radius=0, border_width=3, fg_color="#000000", text_font=("Roman",60), command = self.command, hover_color="#16044a")
            self.name.grid(row=self.row, column=self.column, sticky=tk.NSEW, columnspan=3)


    # loads our model
    class L1Dist(Layer):
        def __init__(self, **kwargs):
            super().__init__()
            
        def call(self, input_embedding, validation_embedding):
            return tf.math.abs(input_embedding - validation_embedding)

    siamese_model = tf.keras.models.load_model(r"/home/lukas/Documents/A level/Computer Science/Coursework/Face recognition deep learning/siamesemodel_2.h5", custom_objects={"L1Dist":L1Dist, "BinaryCrossentropy":tf.losses.BinaryCrossentropy})


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
            if prediction < 0.8:
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

        global match
        global match_colour
        if m.result().numpy() >= 0.875:
            match = "True"
            match_colour = "#00FF00"
        else:
            match = "False"
            match_colour = "#FF0000"

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
    match_label = customtkinter.CTkLabel(window, text=match, corner_radius=0, fg_color=match_colour, text_font=("Roman",60))
    match_label.grid(row=0, column=1, sticky=tk.NSEW, columnspan=3)

    
    compare_button = buttons("Compare Face", 3, 1, button_pressed)
    compare_button.button_create()
    
    # takes screen size
    width, height = pyautogui.size()
    # converts the sizes to fit the label ratio
    width_win_scale = int(width/1.7075)
    height_win_scale = int(height/0.96)
    



    # passport image preview
    passport_text = customtkinter.CTkButton(window, text="Passport Image", text_font = ("Roman", 20), fg_color="#000000", hover=False, border_width=2, corner_radius=0, height = 300)
    passport_text.grid(row=1, column=1, sticky=tk.NSEW)

    # creates a border around the image
    passport_frame = customtkinter.CTkFrame(window, highlightbackground="black", highlightthickness=10)
    passport_frame.grid(row=2, column=1, sticky=tk.NSEW)
    # creates a label for the image to be put in
    passport_label = customtkinter.CTkLabel(window, text="")
    passport_label.grid(row=2, column=1, sticky=tk.NSEW, padx = 2, pady=4)

    
    

    # captured image preview
    input_text = customtkinter.CTkButton(window, text="Compared Faces", text_font = ("Roman", 20), fg_color="#000000", hover=False, border_width=2, corner_radius=0, height = 300)
    input_text.grid(row=1, column=2, sticky=tk.NSEW)

    # creates a border around the image
    input_frame = customtkinter.CTkFrame(window, highlightbackground="black", highlightthickness=10)
    input_frame.grid(row=2, column=2, sticky=tk.NSEW)
    # creates a label for the image to be put in
    input_label = customtkinter.CTkLabel(window, text="") 
    input_label.grid(row=2, column=2, sticky=tk.NSEW, padx=2, pady=4)



    
    cap = cv2.VideoCapture(1)
    cam_label = tk.Label(window)

    cam_label.grid(row=0, column=0, sticky=tk.NSEW, rowspan=4)
    
    def frame_capture():
        # to capture live feed we essentially need to convert to picture frame by frame
        frame = cap.read()[1]

        
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
    
    ANC_PATH = os.path.join("application_data", "input_image")

    while cap.isOpened():
        ret, frame = cap.read()
        # Cut down frame to 250x250px
        frame = frame[120 : 120 + 250, 200 : 200 + 250, :]

        # Collect anchors
        for image in os.listdir(ANC_PATH):
            image_remove = os.path.join(ANC_PATH, image)
            os.remove(image_remove)
        for i in range(300):
            # Create the unique file path
            imgname = os.path.join(ANC_PATH, "{}.jpg".format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        break

    # wrtie the image collected
    for image in os.listdir(ANC_PATH):
        global live_image_cap
        live_image_cap = os.path.join(ANC_PATH, image)

    print("done")

# collect passport image
def collect_passport():
    cap = cv2.VideoCapture(3)
    POS_PATH = os.path.join("application_data", "verification_images")
    NEG_PATH = os.path.join("application_data", "negative_images")

    while cap.isOpened(): 
        ret, frame = cap.read()
        # Cut down frame to 250x250px
        frame = frame[120:120+250,200:200+250, :]
        
        # Collect anchors 
        for image in os.listdir(POS_PATH):
            image_remove = os.path.join(POS_PATH, image)
            os.remove(image_remove)
        for image in os.listdir(NEG_PATH):
            image_remove = os.path.join(NEG_PATH, image)
            os.remove(image_remove)
        for i in range(300):
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, "{}.jpg".format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
            imgname2 = os.path.join(NEG_PATH, "{}.jpg".format(uuid.uuid1()))
            cv2.imwrite(imgname2, frame)
        break
        
    # write the image collected
    for image in os.listdir(POS_PATH):
        global passport_image_cap
        passport_image_cap = os.path.join(POS_PATH, image)

    print("done")



      
compare_faces()
