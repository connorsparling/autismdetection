import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import random
import imutils
import preprocess # custom preprocess.py file
import sys, getopt
import csv
import keras.callbacks.callbacks as callbacks
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix

# Custom keras recall metric for training
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

# Custom keras precision metric for training
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

# Custom keras f1 metric for training
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Custom callback function for collecting intermidiate weight values for analysis
class WeightCollection(callbacks.History):
    # collect "effect" each input has on the final result (sum of total weight values through nodes)
    def analyze_weights(self):
        last_weights = [1.0]
        for layer in self.model.layers[::-1]:
            weights, bias = layer.get_weights()
            sums = np.zeros(weights.shape[0])
            for i in range(weights.shape[0]):
                sums[i] += np.dot(weights[i], last_weights)
            last_weights = sums
        self.my_model.historical_weights.append(last_weights)

    # function called at thebeginning of training
    def on_train_begin(self, logs=None):
        # initialize values
        self.epoch = []
        self.history = {}

    # function called at the end of training epoch
    def on_epoch_end(self, epoch, logs=None):
        # log precision, recall, etc for each epoch
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.analyze_weights()

    def __init__(self, *args, **kwargs):
        self.my_model = kwargs['my_model']
        del kwargs['my_model']
        super().__init__(*args, **kwargs)

class MODEL():
    # load a pre-existing trained model
    def load_model(self):
        # load the model if it was saved previously
        if os.path.exists(self.FILE):
            self.model = tf.keras.models.load_model(self.FILE, custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})

            with open('TrainingHistory.csv', newline='') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                self.val_loss = []
                self.val_accuracy = []
                self.val_f1_m = []
                self.val_precision_m = []
                self.val_recall_m = []
                isFirst = True
                for row in csv_reader:
                    if isFirst:
                        isFirst = False
                        continue
                    self.val_loss.append(float(row[1]))
                    self.val_accuracy.append(float(row[2]))
                    self.val_f1_m.append(float(row[3]))
                    self.val_precision_m.append(float(row[4]))
                    self.val_recall_m.append(float(row[5]))
            csv_file.close()

            with open('WeightHistory.csv', newline='') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                isFirst = True
                self.historical_weights = []
                for row in csv_reader:
                    if isFirst:
                        isFirst = False
                        self.headers = row
                        continue
                    self.historical_weights.append([float(r) for r in row])
            csv_file.close()

            return True
        return False

    
    def save_model(self):
        tf.keras.models.save_model(self.model, self.FILE, True)

        with open('TrainingHistory.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['epoch', 'val_loss', 'val_accuracy', 'val_f1_m', 'val_precision_m', 'val_recall_m'])
            for i in range(len(self.val_loss)):
                csv_writer.writerow([i, self.val_loss[i], self.val_accuracy[i], self.val_f1_m[i], self.val_precision_m[i], self.val_recall_m[i]])
        csv_file.close()

        with open('WeightHistory.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            if self.headers is not None:
                csv_writer.writerow(self.headers)
            for w in self.historical_weights:
                csv_writer.writerow(w)
        csv_file.close()

    def load_data(self):
        X_train, X_test, y_train, y_test, headers = preprocess.load_data(self.subset)
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.headers = headers

    def init_model(self):
        self.model = tf.keras.models.Sequential()

        input_dim = len(self.headers)

        # Setup the model layers: 
        self.model.add(tf.keras.layers.Dense(15, input_dim=input_dim, activation="sigmoid"))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Compile the model with custom metrics
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

    def train_model(self, epochs):
        print("Initializing Model...")
        # load data
        self.load_data()

        self.init_model()

        self.historical_weights = []
        wc = WeightCollection(my_model=self)

        # run and train model
        print("Training Model...")
        self.history = self.model.fit(
            self.X_train, 
            self.y_train,
            batch_size=self.BATCH_SIZE,
            epochs=epochs, 
            verbose=1,
            validation_data=(self.X_test, self.y_test),
            callbacks=[wc]
        )

        self.val_loss = self.history.history['val_loss']
        self.val_accuracy = self.history.history['val_accuracy']
        self.val_f1_m = self.history.history['val_f1_m']
        self.val_precision_m = self.history.history['val_precision_m']
        self.val_recall_m = self.history.history['val_recall_m']

        self.save_model()

    def plot_history(self):
        plt.figure(1, figsize=[8,5])
        plt.plot(self.val_loss)
        plt.plot(self.val_accuracy)
        plt.plot(self.val_f1_m)
        plt.plot(self.val_precision_m)
        plt.plot(self.val_recall_m)
        plt.legend(['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
        plt.xlabel('Epoch Number')
        plt.ylabel('End of Epoch Result')
        plt.title('Validation Results per Epoch')
        plt.show()

        plt.figure(2, figsize=[10,5])
        hist_weight_T = np.transpose(self.historical_weights)
        count = 0
        for row in hist_weight_T:
            col = ''
            if self.subset == 3 and self.headers[count][0] == 'A':
                if self.headers[count][-1] == '1':
                    col = '--'
                elif self.headers[count][-1] == '0':
                    col = ':'
            plt.plot(row, col)
            count += 1

        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(self.headers, loc='upper right', prop=fontP)
        plt.xlim((0, int(len(self.historical_weights)*1.25)))
        plt.xlabel('Epoch Number')
        plt.ylabel('End of Epoch Result')
        plt.title('Validation Results per Epoch')
        #plt.yscale('log')
        plt.show()

    def test_model(self):
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print()
        print("Loss:      {:.4f}".format(score[0]))
        print("Accuracy:  {:.2f}%".format(score[1]*100))
        print("F1 Score:  {:.2f}%".format(score[2]*100))
        print("Precision: {:.2f}%".format(score[3]*100))
        print("Recall:    {:.2f}%".format(score[4]*100))

        train_predictions = self.model.predict(self.X_train, verbose=0)
        train_pred = [(1 if p > 0.5 else 0) for p in train_predictions]
        print('\nTrain Confusion Matrix')
        print(confusion_matrix(self.y_train, train_pred))

        test_predictions = self.model.predict(self.X_test, verbose=0)
        test_pred = [(1 if p > 0.5 else 0) for p in test_predictions]
        print('\nTest Confusion Matrix')
        print(confusion_matrix(self.y_test, test_pred))
        # print('Classification Report')
        # target_names = ['Cats', 'Dogs', 'Horse']
        # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

    def analyze_weights(self):
        last_weights = [1.0]
        for layer in self.model.layers[::-1]:
            weights, bias = layer.get_weights()
            sums = np.zeros(weights.shape[0])
            for i in range(weights.shape[0]):
                sums[i] += np.dot(weights[i], last_weights)
            last_weights = sums
        return np.abs(last_weights)

    def feature_importance(self):
        last_weights = self.analyze_weights()
        print("\nFeature Importance:")
        relevance = [[y, x] for y, x in sorted(zip(last_weights,self.headers))][::-1]
        for i in range(len(relevance)):
            space = " "*(30 - len(str(relevance[i][1])))
            print("{}{}{:.4f}".format(relevance[i][1], space, relevance[i][0]))

    def __init__(self, batch=50):
        self.FILE = "FORWARD.h5"
        self.BATCH_SIZE = batch
        self.subset = 0

def main(argv):
    filename = None
    plot = False
    epochs = None
    importance = False
    subset = 0
    try:
        opts, args = getopt.getopt(argv,"hdipt:f:s:",["file=", "plot", "default", "train=", "importance", "subset="])
    except getopt.GetoptError:
        print("INCORRECT FORMAT: \"model.py [-f <output_file> | -d] [-t <epochs>] [-p] [-i] [-s <which_subset>]\"")
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
            plot = True
        elif opt in ("-t", "--train"):
            epochs = int(arg)
        elif opt in ("-i", "--importance"):
            importance = True
        elif opt in ("-s", "--subset"):
            subset = int(arg)

    model = MODEL()
    model.subset = subset

    if epochs is not None:
        model.train_model(epochs)
    else: 
        model.load_data()
        if filename is not None:
            if filename != "default":
                if not filename.endswith(".h5"):
                    print("Please specify a file with the '.h5' file type")
                    sys.exit()
                model.FILE = filename
            if not model.load_model():
                print("Model failed to load")
                sys.exit()

    # test model
    model.test_model()
    
    if importance:
        model.feature_importance()
    
    if plot:
        model.plot_history()
        
if __name__ == '__main__':
	main(sys.argv[1:])