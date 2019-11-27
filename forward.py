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
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class MODEL():
    def load_model(self):
        # load the model if it was saved previously
        if os.path.exists(self.FILE):
            self.model = tf.keras.models.load_model(self.FILE)
            return True
        return False

    def init_model(self):
        self.model = tf.keras.models.Sequential()

        input_dim = 27

        # Model 1: 99.05% accuracy after 36 epochs
        # self.model.add(tf.keras.layers.Dense(36, input_dim=input_dim, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(12, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Model 2: 99.05% accuracy with 20 epochs and 99.53% accuracy with 33 epochs
        # self.model.add(tf.keras.layers.Dense(50, input_dim=input_dim, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(20, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Model 3: 
        # self.model.add(tf.keras.layers.Dense(1000, input_dim=input_dim, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(100, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Model 4: 
        # self.model.add(tf.keras.layers.Dense(15, input_dim=input_dim, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Model 5: 
        self.model.add(tf.keras.layers.Dense(30, input_dim=input_dim, activation="relu"))
        self.model.add(tf.keras.layers.Dense(10, input_dim=input_dim, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

    def train_model(self, epochs):
        print("Initializing Model...")
        self.init_model()

        # load data
        X_train, X_test, y_train, y_test, headers = preprocess.load_data()
        self.headers = headers

        # run and train model
        print("Training Model...")
        self.model.fit(
            np.array(X_train), 
            np.array(y_train),
            batch_size=self.BATCH_SIZE,
            epochs=epochs, 
            verbose=1,
            validation_data=(np.array(X_test), np.array(y_test))
        )

        # test model
        self.test_model(np.array(X_test), np.array(y_test))

        # save the file for use in future sessions
        tf.keras.models.save_model(self.model, self.FILE, True)

    def test_model(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print("Loss: " + str(score[0]))
        print("Accuracy: " + str(score[1]))
        print("F1 Score: " + str(score[2]))
        print("Precision: " + str(score[3]))
        print("Recall: " + str(score[4]))
    
    def plot_model(self, filename):
        tf.keras.utils.plot_model(self.model, show_shapes=True, to_file=filename)

    def analyze_weights(self):
        last_weights = [1.0]
        for layer in self.model.layers[::-1]:
            weights, bias = layer.get_weights()
            sums = np.zeros(weights.shape[0])
            for i in range(weights.shape[0]):
                sums[i] += np.dot(weights[i], last_weights)
            last_weights = sums

        print("\nRelevance to Autism Classification:")
        relevance = [x for _,x in sorted(zip(last_weights,self.headers))][::-1]
        for i in range(len(relevance)):
            print("{}: {}".format(i, relevance[i]))

    def __init__(self, batch=50):
        self.FILE = "FORWARD.h5"
        self.BATCH_SIZE = batch

def main(argv):
    filename = None
    plot_filename = None
    epochs = None
    try:
        opts, args = getopt.getopt(argv,"ht:df:p:",["file=", "plot=", "default", "train="])
    except getopt.GetoptError:
        print("INCORRECT FORMAT: \"model.py [-f <output_file> | -d] [-t <epochs>] [-p <plot_image_file>]\"")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("model.py [-f <output_file> | -d] [-t]")
            sys.exit()
        elif opt in ("-f", "--file"):
            filename = arg
        elif opt in ("-d", "--default"):
            filename = "default"
        elif opt in ("-p", "--plot"):
            plot_filename = arg
        elif opt in ("-t", "--train"):
            epochs = int(arg)

    model = MODEL()

    if epochs is not None:
        model.train_model(epochs)
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
    
    if plot_filename is not None:
        model.plot_model(plot_filename)
    
    model.analyze_weights()
        
if __name__ == '__main__':
	main(sys.argv[1:])