import cv2
import os
import uuid

def collect_passport():
    cap = cv2.VideoCapture(3)
    POS_PATH = os.path.join("application_data", "verification_images")

    while cap.isOpened(): 
        ret, frame = cap.read()
        # Cut down frame to 250x250px
        frame = frame[120:120+250,200:200+250, :]
        
        # Collect anchors 
        for image in os.listdir(POS_PATH):
            image_remove = os.path.join(POS_PATH, image)
            os.remove(image_remove)
        for i in range(300):
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, "{}.jpg".format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        break
        
    # write the image collected
    for image in os.listdir(POS_PATH):
        global passport_image_cap
        passport_image_cap = os.path.join(POS_PATH, image)
        
    print("done")

collect_passport()