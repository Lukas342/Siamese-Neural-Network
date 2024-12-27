# Face Recognition with Siamese Network

This project implements a face recognition system using a Siamese neural network. The Siamese network is trained to differentiate between pairs of images, determining whether they belong to the same person or not. This approach is particularly useful for tasks such as face verification and identification.

## Project Structure

- `ai_model/Face recognition with siamese network.ipynb`: Jupyter notebook containing the implementation and training of the Siamese network.
- `application/compare_faces.py`: Script to compare faces using the trained model.
- `application/application_data`: Contains the input image and passport image from `compare_faces.py` and `collect_passport.py`
- `application/re_train.py`: Script to retrain the model.
- `data/`: Directory containing positive, negative, and anchor images for training.

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/Siamese-neural-network.git
    cd Siamese-neural-network
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the Labelled Faces in the Wild (LFW) dataset:**
    ```sh
    wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
    tar -xf lfw.tgz
    ```

## Training the Model

1. **Run the Jupyter notebook:**
    Open `ai_model/Face recognition with siamese network.ipynb` and follow the steps to collect, preprocess the images, build the model, and train it.

2. **Save the trained model:**
    The trained model will be saved as `siamese_model.h5`.

Alternatively, if you only want to create the model without worrying about data collection, you can use the retrain option as mentioned below.

## Using the Model

1. **Access the main menu:**
    Use the `main.py` script to access different functionalities such as comparing faces, collecting passport photo and retraining the model:
    ```sh
    python application/main.py
    ```

2. **Collect passport photo:**
    From the main menu, select the option to collect passport photo. Press `p` to capture the image.

3. **Compare faces:**
    From the main menu, select the option to compare two faces. Click compare faces to compare the current webcam to the passport photo previously collected.

4. **Retrain the model:**
    From the main menu, select the option to retrain the model and follow the prompts to provide the new data, or retrain from `compare_faces.py`


## Acknowledgements

- The project uses the Labelled Faces in the Wild (LFW) dataset.
- The Haar Cascade model for face detection is provided by OpenCV.

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/yourusername/Siamese-neural-network/issues).
