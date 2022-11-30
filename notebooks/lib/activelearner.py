import os
import zipfile
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
os.environ['KAGGLE_USERNAME'] = 'dumpstertrash'
os.environ['KAGGLE_KEY'] = 'fc29a78761d58a50c7b50d71f3f3a3f2'
from kaggle.api.kaggle_api_extended import KaggleApi

from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, categorical_crossentropy
from sklearn import metrics
from tensorflow.keras.models import load_model

from lib.dataset import Dataset


class ActiveLearner():
    
    def __init__(self, dataset, architecture:str="vgg16"):
        self.architecture = architecture
        self.model = None  # the underlying model

        self.dataset = dataset  # the dataset
    
        self.used = [0, 0, 0] #  amount of imgs used per class
        self.acc = 0.0  # the accuracy of the model on the test-set

        self.X_pool = None # the pool of data to query from
        self.X_train = None  # the train data
        self.X_test = None  # the test data
        self.y_pool = None  # the labels for the pool
        self.y_train = None  # the train labels
        self.y_test = None  # the test labels

        self.p_test = None  # the predicted probabilities

        self.epochs=10  # the amount of epochs to train (the amount of times to go though the train data set)
        self.batch_size=64  # the batch_size used during training        
        
        # create the internal model
        self.new_model()
        
        self.X_pool, self.X_test, self.y_pool, self.y_test = self.dataset.split()
        
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return f"<|AL|{self.architecture}|used|{sum(self.used)}|{self.used}|acc|{self.acc}|>"
    
    def new_model(self):
        if self.architecture == "vgg16":
            self.model = VGG16(include_top=True, weights=None, classes=3)
            self.model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
        else:
            raise ValueError(f"architecture {self.architecture} not supported")
        # new model is untrained, so no samples are used, and accuracy is untested (=0)
        self.used = [0, 0, 0]
        self.acc = 0.0
        return self
    
    def with_params(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        return self
    
    def test(self):
        """
        Test the model on the test-set
        """
        if any([var is None for var in [self.model, self.X_test, self.y_test]]):
            raise ValueError(f"model not properly instantiated")
        print(f"model: testing...")
        self.p_test = np.array(list(map(self.dataset.from_classlist, self.model.predict(self.X_test, verbose=0))))
        count = 0
        for i in range(self.p_test.shape[0]):
            if self.p_test[i] == self.y_test[i]:
                count += 1
        self.acc = count / self.p_test.shape[0]
        return self
    
    def train(self):
        if any([var is None for var in [self.model, self.X_train, self.y_train]]):
            raise ValueError(f"model not properly instantiated")
        print(f"model: training...")
        for x in self.y_train:
            self.used[x] += 1
        self.model.fit(
            x=self.X_train,
            y=np.array(list(map(self.dataset.to_classlist, self.y_train))), 
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self
    
    def select_samples(self, query_strategy, n):
        print(f"model: selecting {n} imgs...")
        idxs = query_strategy(self.model, self.X_pool, n)
        # move the selected images to the training set
        self.X_train = self.X_pool[idxs]
        self.y_train = self.y_pool[idxs]
        # remove the selected images from the pool
        self.X_pool = np.delete(self.X_pool, idxs, axis=0)
        self.y_pool = np.delete(self.y_pool, idxs, axis=0)
        return self
    
    def train_al(self, max_imgs:int, strategy, query_size:int=32):
        """
        Train the model on the active learning set
        """
        if any([var is None for var in [self.model, self.X_pool, self.y_pool]]):
            raise ValueError(f"model not properly instantiated")
    
        iterations = ceil(max_imgs / query_size)

        for i in range(iterations):
            # can we train on the query size, or is less remaining
            imgs = np.min([query_size, (max_imgs - query_size*i)])
            print(f"iteration {i+1}/{iterations}, training on {imgs} images (max {max_imgs})")
            # ask the query strategy for query_size indexes to label, or the remaining amount of images
            idxs = strategy(self.model, X_pool, imgs)
            # get new model
            self.new_model()  # clears what images were used
            # move the selected images to the training set
            self.X_train = np.append(self.X_train, X_pool[idxs], axis=0)
            self.y_train = np.append(self.y_train, y_pool[idxs], axis=0)
            # remove the selected images from the pool
            self.X_pool = np.delete(self.X_pool, idxs, axis=0)
            self.y_pool = np.delete(self.y_pool, idxs, axis=0)
            # train the model
            self.train()  # fills in which images were used

        return self

    def clean(self):
        self.model = None
        self.X_pool = None
        self.X_train = None
        self.X_test = None
        self.y_pool = None
        self.y_train = None
        return self
    
    def clean_full(self):
        self.clean()
        self.y_test = None  # used for confusion matrix
        self.p_test = None  # used for confusion matrix
        self.dataset = None  # used for histogram used classes
        return self
    

    def confusion_matrix(self, loc='confusion_matrix.png'):
        """
        Creates a confusion matrix of the model.
        """
        if any([var is None for var in [self.p_test, self.y_test]]):
            raise ValueError(f"model not properly instantiated")
        p_labels = np.array(list(map(self.dataset.to_label, self.p_test)))
        y_labels = np.array(list(map(self.dataset.to_label, self.y_test)))
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_labels, p_labels)
        disp.figure_.suptitle(f"Confusion Matrix (acc: {self.acc})")
        plt.savefig(loc)

    def histogram_used_classes(self, loc='used_classes.png'):
        """
        Creates a histogram of the used classes.
        """
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
