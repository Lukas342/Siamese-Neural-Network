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

    siamese_model = tf.keras.models.load_model("siamese_model.h5", custom_objects={"L1Dist":L1Dist, "BinaryCrossentropy":tf.losses.BinaryCrossentropy})

    # image paths
    POS_PATH = os.path.abspath("data\\positive")
    NEG_PATH = os.path.abspath("data\\negative")
    ANC_PATH = os.path.abspath("data\\anchor")

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
    checkpoint_dir = os.path.abspath("../ai_model/training_checkpoints")
    # ensures that all our checkpoints have the prefix of ckpt
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # saves our the model and optimiser at the time we run the checkpoint class
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    # compiles our function into a callable TensorFlow graph
    @tf.function
    def train_step(batch):
        # allows us to capture our gradient from the model, records the operations for automatic differentiation
        # in short tape will record data, so that when we call it, we can take the data from the whole model
        # records all of our operations
        with tf.GradientTape() as tape:
            # get anchor and positive/negative images
            # remember when we defined our train data, we had a list containing [anchor, positive/negative, label]
            x = batch[:2]
            # gets the label
            y = batch[2]

            # forward pass
            # passes our data through the siamese model to make a prediction
            y_pred = siamese_model(x, training=True)
            # calculate loss
            # to calculate our loss we first pass through our y true value, so our label. Then we pass through our predicted value
            # the smaller the loss the closer our prediction is to the true labels (y_pred). 
            loss = binary_loss(y, y_pred)
        

            # calculate gradients
            # calculates all of the gradients in respect to our loss for all of our trainable variables
            # calculates all the graidients for our different wieghts within our specific model in respect to our loss
            grad = tape.gradient(loss, siamese_model.trainable_variables)

            # calculate updated weights and apply to siamese model
            # the optimiser is calculating and propagating the new weights using Adam's optimisation algorithm, a variant of gradient desecent
            # applying our learning rate and slightly reducing the loss by changing the weights to be closer to the optimiser.
            # 
            opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

        return loss


    def train(data, EPOCHS):
        # loop through the EPOCHS
        for epoch in range(1, EPOCHS + 1):
            print(f"\n Epoch {epoch}/{EPOCHS}")
            progress_bar = tf.keras.utils.Progbar(len(data))
            
            # loop through each batch
            for index, batch in enumerate(data):
                # applying our train step function to a single batch
                train_step(batch)
                # updating our progress bar
                progress_bar.update(index + 1)

            # save checkpoint every 10 epochs
            if epoch % 10 == 0: 
                checkpoint.save(file_prefix=checkpoint_prefix)


    EPOCHS = 100 # num times we will run through the training data
    train(train_data, EPOCHS)

    siamese_model.compile(optimizer=opt, loss=binary_loss, metrics=['accuracy']) 

    siamese_model.save("siamese_model.h5")