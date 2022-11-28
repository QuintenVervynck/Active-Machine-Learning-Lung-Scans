import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
os.environ['KAGGLE_USERNAME'] = 'dumpstertrash'
os.environ['KAGGLE_KEY'] = 'fc29a78761d58a50c7b50d71f3f3a3f2'
from kaggle.api.kaggle_api_extended import KaggleApi

from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from sklearn import metrics
from tensorflow.keras.models import load_model

from lib.dataset import Dataset


class Model():
    
    def __init__(self, architecture:str="vgg16", dataset=None):
        self.architecture = architecture
        self.model = None  # the underlying model

        self.dataset = dataset  # the dataset
    
        self.used = [0, 0, 0] #  amount of imgs used per class
        self.acc = 0.0  # the accuracy of the model on the test-set

        self.X_train = None  # the train data
        self.X_valid = None  # the validation data
        self.X_test = None  # the test data
        self.y_train = None  # the train labels
        self.y_test = None  # the test labels
        self.y_valid = None  # the validation labels

        self.size = None  # the size of the training set

        self.p_test = None  # the predicted probabilities

        self.epochs=10  # the amount of epochs to train (the amount of times to go though the train data set)
        self.batch_size=64  # the batch_size used during training        
        
        # create the internal model
        if architecture == "vgg16":
            self.model = VGG16(weights=None, classes=3)
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])
        else:
            raise ValueError(f"model architecture '{architecture}' does not exist.")
        
        # prepare the data
        if self.dataset is None:
            self.dataset = Dataset()
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self.dataset.split()
        self.size = self.X_train.shape[0]
        
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return f"<Model {self.architecture}: used {sum(self.used)} {self.used}, acc {self.acc}>"
    
    def limit_training_data(self, size):
        self.X_train = self.X_train[:size]
        self.y_train = self.y_train[:size]
        self.size = size
        return self
    
    def limit_all_data(self, size):
        self.X_train = self.X_train[:size]
        self.y_train = self.y_train[:size]
        self.X_valid = self.X_valid[:size]
        self.y_valid = self.y_valid[:size]
        self.X_test = self.X_test[:size]
        self.y_test = self.y_test[:size]
        self.size = size
        return self
    
    def set_epochs(self, epochs):
        self.epochs = epochs
        return self
       
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    
    def test(self):
        if any([var is None for var in [self.model, self.X_test, self.y_test]]):
            raise ValueError(f"model not properly instantiated")
        self.p_test = np.array(list(map(self.dataset.from_classlist, self.model.predict(self.X_test))))
        count = 0
        for i in range(self.p_test.shape[0]):
            if self.p_test[i] == self.y_test[i]:
                count += 1
        self.acc = count / self.p_test.shape[0]
        return self
    
    def train(self):
        if any([var is None for var in [self.model, self.X_train, self.X_valid, self.y_train, self.y_valid]]):
            raise ValueError(f"model not properly instantiated")
        for x in self.y_train:
            self.used[x] += 1
        self.model.fit(
            x=self.X_train,
            y=np.array(list(map(self.dataset.to_classlist, self.y_train))), 
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.X_valid, np.array(list(map(self.dataset.to_classlist, self.y_valid))))
        )
        return self
    
    def clean(self):
        self.model = None
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.size = None
        return self
    
    def clean_full(self):
        self.clean()
        self.y_test = None  # used for confusion matrix
        self.p_test = None  # used for confusion matrix
        self.dataset = None  # used for histogram used classes
        return self
    
    def confusion_matrix(self, loc='confusion_matrix.png'):
        if any([var is None for var in [self.p_test, self.y_test]]):
            raise ValueError(f"model not properly instantiated")
        p_labels = np.array(list(map(self.dataset.to_label, self.p_test)))
        y_labels = np.array(list(map(self.dataset.to_label, self.y_test)))
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_labels, p_labels)
        disp.figure_.suptitle(f"Confusion Matrix (acc: {self.acc})")
        plt.savefig(loc)

    def histogram_used_classes(self, loc='used_classes.png'):
        # Show histogram of used classes
        if any([var is None for var in [self.dataset, self.used]]):
            raise ValueError(f"model not properly instantiated")
        labels = self.dataset.names
        counts = self.used
        ticks = range(len(counts))
        plt.bar(ticks, counts, align='center')
        plt.xticks(ticks, labels)
        plt.savefig(loc)
        plt.show()
