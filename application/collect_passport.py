import cv2
import os
import uuid
   

def collect_passport():

    POS_PATH = os.path.abspath("application\\application_data\\verification_images")
    # Establish a connection to the webcam
    cap = cv2.VideoCapture(1)
    while cap.isOpened(): 
        ret, frame = cap.read()
    
        # Cut down frame to 250x250px
        frame = frame[120:120+250,200:200+250, :]
        
        # Collect anchors 
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
            break
        
        # Show image back to screen
        cv2.imshow('Image Collection', frame)

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()

collect_passport()