import cv2
import os
import uuid

def collect_live():
    global cap
    
    ANC_PATH = os.path.join("application_data", "input_image")

    while cap.isOpened():
        ret, frame = cap.read()
        # Cut down frame to 250x250px
        frame = frame[120 : 120 + 250, 200 : 200 + 250, :]

        # Collect anchors
        for image in os.listdir(os.path.join("application_data", "input_image")):
            image_remove = os.path.join("application_data", "input_image", image)
            os.remove(image_remove)
        for i in range(300):
            # Create the unique file path
            imgname = os.path.join(ANC_PATH, "{}.jpg".format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        break


    for image in os.listdir(os.path.join("application_data", "input_image")):
        global live_image_cap
        live_image_cap = os.path.join(ANC_PATH, image)

    print("done")