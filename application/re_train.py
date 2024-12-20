import os

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

import shutil
import add_database
def re_train():
    # loads our model
    class L1Dist(Layer):
        def __init__(self, **kwargs):
            super().__init__()
            
        def call(self, input_embedding, validation_embedding):
            return tf.math.abs(input_embedding - validation_embedding)

    siamese_model = tf.keras.models.load_model("siamesemodel_2.h5", custom_objects={"L1Dist":L1Dist, "BinaryCrossentropy":tf.losses.BinaryCrossentropy})

    # image paths
    ANC_PATH = os.path.join("data", "anchor")
    POS_PATH = os.path.join("data", "positive")
    NEG_PATH = os.path.join("data", "negative")


    # calculates how many images are in the file
    count = 0
    for image in os.listdir(POS_PATH):
        count += 1

    anchor = tf.data.Dataset.list_files(ANC_PATH+"/*.jpg").take(count)
    positive = tf.data.Dataset.list_files(POS_PATH+"/*.jpg").take(count) 
    negative = tf.data.Dataset.list_files(NEG_PATH+"/*.jpg").take(count)

    def preprocess(file_path):
        # reading img
        byte_img = tf.io.read_file(file_path)
        # using tf decode image to load it in
        img = tf.io.decode_jpeg(byte_img)
        # resizes img
        img = tf.image.resize(img, (100,100))
        # scales image to be between 0 and 1
        img = img / 255.0
        return img

    # combines the anchor and positve/negative image. Adds 1.0/0.0 depending if same face
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    def preprocess_twin(input_img, validation_img, label):
        return (preprocess(input_img), preprocess(validation_img), label)

    #### Build dataLoader Pipeline ####
    # maps our data
    data = data.map(preprocess_twin)
    # caching our images so we can access them faster
    data = data.cache()
    # shuffles all our data,
    data = data.shuffle(buffer_size=1024)


    #### Training Partition ####
    # takes 70% of images for training data
    train_data = data.take(round(len(data)*.7))
    train_data = train_data.batch(16)
    # starts preprocessing the next set of images so that we don"t bottle neck our next set images
    train_data = train_data.prefetch(8)



    # the loss will be used later to be able to calculate our loss (1 or 0)
    binary_loss = tf.losses.BinaryCrossentropy()
    # improves speed and performance
    opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

    # defined our checkpoint dir
    checkpoint_dir = "./training_checkpoints"
    # ensures that all our checkpoints have the prefix of ckpt
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # saves our the model and optimiser at the time we run the checkpoint class
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    @tf.function # compiles our function into a callable TensorFlow graph
    def train_step(batch):
        # allows us to capture our gradient from the model, records the operations for automatic differentiation
        with tf.GradientTape() as tape:
            x = batch[:2] # get anchor and positive/negative images
            y = batch[2] # takes the label

            # passes our data through the siamese model to make a prediction
            y_pred = siamese_model(x, training=True)
            # calculates the loss
            loss = binary_loss(y, y_pred)
        # calculates all of the gradients in respect to our loss for all of our trainable variables   
        grad = tape.gradient(loss, siamese_model.trainable_variables)
        # calculate updated weights and apply to siamese model
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

        return loss


    def train(data, EPOCHS):
        for epoch in range(1, EPOCHS+1):
            print(f"\n Epoch {epoch}/{EPOCHS}")
            progress_bar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            train_step(batch)
            progress_bar.update(idx+1)

        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


    EPOCHS = 50 # num times we will run through the training data
    #train(train_data, EPOCHS)

    siamese_model.save("siamese_model.h5")


re_train()