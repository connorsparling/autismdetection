import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import random
import imutils
import preprocess
import sys, getopt
import csv

class MODEL():
    def load_model(self):
        # load the model if it was saved previously
        if os.path.exists(self.FILE):
            self.model = tf.keras.models.load_model(self.FILE)
            return True
        return False

    def init_model(self):
        self.model = tf.keras.models.Sequential()

        # Model 1: 99.05% accuracy after 36 epochs
        # self.model.add(tf.keras.layers.Dense(36, input_dim=27, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(12, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Model 2: 99.05% accuracy with 20 epochs and 99.53% accuracy with 33 epochs
        # self.model.add(tf.keras.layers.Dense(50, input_dim=27, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(20, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Model 3: 
        # self.model.add(tf.keras.layers.Dense(1000, input_dim=27, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(100, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Model 4: 
        self.model.add(tf.keras.layers.Dense(15, input_dim=27, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        print("Initializing Model...")
        self.init_model()

        # load data
        X_train, X_test, y_train, y_test = preprocess.load_data()

        # run and train model
        print("Training Model...")
        self.model.fit(
            np.array(X_train), 
            np.array(y_train),
            batch_size=self.BATCH_SIZE,
            epochs=100, 
            verbose=1,
            validation_data=(np.array(X_test), np.array(y_test))
        )

        # test model
        score = self.model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # save the file for use in future sessions
        tf.keras.models.save_model(self.model, self.FILE, True)

        count = 0
        with open('layer_weights.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for layer in self.model.layers:
                count += 1
                print("Layer {} Shape: {}".format(count, [len(w) for w in layer.get_weights()]))
                for l in layer.get_weights():
                    csv_writer.writerow(l)#layer.get_weights())
        csv_file.close()


    def detect(self, image):
        # Save colour image to show to user
        image = self.prep_image(image)
        image = image.reshape(1, self.IMG_WIDTH, self.IMG_HEIGHT, 1)
        # Cast to float to handle error
        image = tf.cast(image, tf.float32)
        prediction = self.model.predict(image)
        # Convert prediction from one hot to category
        index = tf.argmax(prediction[0], axis=0)
        return self.CATEGORIES[index]

    def __init__(self, batch=50):
        self.FILE = "FORWARD.h5"
        self.BATCH_SIZE = batch

def main(argv):
    filename = None
    train = False
    try:
        opts, args = getopt.getopt(argv,"htdf:",["file=", "default", "train"])
    except getopt.GetoptError:
        print("INCORRECT FORMAT: \"model.py [-f <output_file> | -d] [-t]\"")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("model.py [-f <output_file> | -d] [-t]")
            sys.exit()
        elif opt in ("-f", "--file"):
            filename = arg
        elif opt in ("-d", "--default"):
            filename = "default"
        elif opt in ("-t", "--train"):
            train = True

    model = MODEL()

    if train:
        model.train_model()
    else: 
        if filename is not None:
            if filename != "default":
                if not filename.endswith(".h5"):
                    print("Please specify a file with the '.h5' file type")
                    sys.exit()
                model.FILE = filename
            if not model.load_model():
                print("Model failed to load")
                sys.exit()
        
if __name__ == '__main__':
	main(sys.argv[1:])