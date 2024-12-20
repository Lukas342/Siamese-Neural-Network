import cv2
import tkinter as tk
import customtkinter
import customtkinter
from PIL import ImageTk, Image
import pyautogui
import os
import uuid

window = customtkinter.CTk()
width, height = pyautogui.size()
# converts the sizes to fit the label ratio
width_win_scale = int(width / 1.7075)
height_win_scale = int(height / 0.96)


cam_label = tk.Label(window)
cam_label.grid(row=0, column=0, sticky=tk.NSEW, rowspan=4)
x = 1
cap = cv2.VideoCapture(1)


def frame_capture():
    global x
    # to capture live feed we essentially need to convert to picture frame by frame
    frame = cap.read()[1]

    frame = cv2.resize(frame, (width_win_scale, height_win_scale))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    img_tk = ImageTk.PhotoImage(image=img)

    cam_label.img_tk = img_tk
    cam_label.configure(image=img_tk)
    x += 1
    if x == 100:
        collect_passport()

    cam_label.after(10, frame_capture)


frame_capture()


def collect_passport():
    global cap
    POS_PATH = os.path.join("application_data", "verification_images")

    while cap.isOpened(): 
        ret, frame = cap.read()
        # Cut down frame to 250x250px
        frame = frame[120:120+250,200:200+250, :]
        
        # Collect anchors 
        for image in os.listdir(os.path.join("application_data", "verification_images")):
            image_remove = os.path.join("application_data", "verification_images", image)
            os.remove(image_remove)
        for i in range(300):
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, "{}.jpg".format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        break
        
    # write the image collected
    for image in os.listdir(os.path.join("application_data", "verification_images")):
        passport_image_cap = os.path.join(POS_PATH, image)
            
    print("done")


window.mainloop()
