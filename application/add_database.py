import shutil
import os
def add_database():
    # image paths
    IMG_PATH = os.path.join("application_data", "input_image", "input_img.jpg")
    POS_IMG = os.path.join("data", "positive", "input_img.jpg")
    VER_PATH = os.path.join("application_data", "verification_images")

    # copying the files
    shutil.copyfile(IMG_PATH, POS_IMG)
    for image in os.listdir(VER_PATH):
        VER_IMG = os.path.join(VER_PATH, image)
    
    ANC_IMG = os.path.join("data", "anchor", image)
    shutil.copyfile(VER_IMG, ANC_IMG)