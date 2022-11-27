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


class Robot():  
    dataset = None  # the dataset object
    
    X_train = None  # the train data
    X_valid = None  # the validation data
    X_test = None  # the test data
    y_train = None # the train labels
    y_valid = None # the validation labels
    y_test = None # the test labels

    size = None  # the size of the dataset
    
    model = None # the machine learning model
    
    used = None  # array with the indices of the datasamples used during training
    
    def __init__(self, mode:str="quick-testrun"):

        if mode == "tf-gpu-check":
            print("ROBOT: TF GPU CHECK")
            import tensorflow as tf
            print(tf.config.list_physical_devices())
        elif mode == "full-learn":
            print("ROBOT: FULL LEARN")
            self.model = self.prep_model_vgg()
            print("ROBOT: prep data")
            self.dataset = Dataset()
            self.prep_data()
            print("ROBOT: train")
            self.train(self.X_train, self.y_train, 10, 64)
            print("ROBOT: saving model")
            self.model.save("models/full_learn.h5")
            print("ROBOT: test")
            self.test()
            self.confusion_matrix("full-learn-confusion-matrix.png")
            self.histogram_used_classes("full-learn-histogram-used-classes.png")
        elif mode == "full-learn-load":
            print("ROBOT: FULL LEARN LOAD")
            print("ROBOT: prep data")
            self.dataset = Dataset()
            self.prep_data()
            print("ROBOT: load model")            
            self.model = load_model('models/full_learn.h5')
            print("ROBOT: test")
            self.test()
            self.confusion_matrix()
        elif mode == "al-learn":
            print("ROBOT: AL LEARN")
            self.model = self.prep_model_vgg()
            print("ROBOT: prep data")
            self.dataset = Dataset()
            self.prep_data()
            print("ROBOT: train")
            self.train_AL(10,64)
            print("ROBOT: saving model")
            self.model.save("models/al_learn.h5")
            print("ROBOT: test")
            self.test()
            self.confusion_matrix("al-learn-confusion-matrix.png")
            #self.histogram_used_classes("al-learn-histogram-used-classes.png")
        elif mode =="al-learn-load":
            print("ROBOT: AL LEARN LOAD")
            print("ROBOT: prep data")
            self.dataset = Dataset()
            self.prep_data()
            print("ROBOT: load model")            
            self.model = load_model('models/al_learn.h5')
            print("ROBOT: test")
            self.test()
            self.confusion_matrix()
        elif mode == "full-learn-acc-graph":
            print("ROBOT: FULL LEARN ACC GRAPH")
            print("ROBOT: prep data")
            self.dataset = Dataset()
            self.prep_data()
            print("ROBOT: create models")
            steps = [i for i in range(50, 1050, 50)] + [i for i in range(1100, 2001, 100)]
            models = [self.prep_model_vgg() for _ in steps]
            accs = []
            used = []
            print("ROBOT: train")
            for i, model in enumerate(models):
                print(f"ROBOT: training model {i+1}/{len(models)}")
                self.model = model
                used.append(self.train(self.X_train[:steps[i]], self.y_train[:steps[i]],3 ,64))
                accs.append(self.test())
            self.used = used
            self.accs = accs
            self.accuracy_graph(used, accs, "full-learn-acc-graph.png")
            
        else:
            print(f"ROBOT: UNKNOWN MODE '{mode}'")


    def prep_data(self, train_size=0, valid_size=0, test_size=0):
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self.dataset.split()
        self.size = self.X_train.shape[0]
    
    def limit_size(self, size):
        self.X_train = self.X_train[:size]
        self.y_train = self.y_train[:size]
        self.X_valid = self.X_valid[:size]
        self.y_valid = self.y_valid[:size]
        self.X_test = self.X_test[:size]
        self.y_test = self.y_test[:size]
        self.size = size
    
    def prep_model_vgg(self):
        model = VGG16(weights=None, classes=3)
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])
        return model

    
    def train(self, X_train, y_train, epochs, batch_size):
        used = [0,0,0]
        for x in y_train:
            used[x] += 1
        self.model.fit(
            x=X_train,
            y=np.array(list(map(self.dataset.to_classlist, y_train))), 
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_valid, np.array(list(map(self.dataset.to_classlist, self.y_valid))))
        )
        print(f"Used samples: {used[0]} {used[1]} {used[2]}")
        self.used = used
        return used



    def train_AL(self, epochs, batch_size):
        iterations = 20
        query_size = 128

        X = self.X_train
        y = np.array(list(map(self.dataset.to_classlist, self.y_train)))
        y_valid_classlist = np.array(list(map(self.dataset.to_classlist, self.y_valid)))
        models = [VGG16(weights=None, classes=3) for _ in range(iterations)]
        [model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy']) for model in models]
        
        # train first model
        idxs = [i for i in range(query_size)]
        X_train = X[idxs]
        y_train = y[idxs]
        X = np.delete(X, idxs, axis=0)
        y = np.delete(y, idxs, axis=0)
        models[0].fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_valid, y_valid_classlist))
        p = models[0].predict(X)
        self.model = models[0]
        self.test()
        self.confusion_matrix(f"al-learn-confusion-matrix-{0}.png")

        for i in range(1, iterations):
            print(f"iteration {i}, {X.shape[0]} samples left")
            
            idxs = np.max(p, axis=1).argsort()[:query_size]
            X_train = np.concatenate((X_train, X[idxs]))
            y_train = np.concatenate((y_train, y[idxs]))
            X = np.delete(X, idxs, axis=0)
            y = np.delete(y, idxs, axis=0)
            models[i].fit(
                x=X_train,
                y=y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(self.X_valid, y_valid_classlist))
            p = models[i].predict(X)
            self.model = models[i]
            self.test()
            self.confusion_matrix(f"al-learn-confusion-matrix-{i}.png")
            
        self.model = models[-1]
        print(f"Used samples: {iterations*query_size}")

        
    def test(self):
        self.p_test = np.array(list(map(self.dataset.from_classlist, self.model.predict(self.X_test))))
        count = 0
        for i in range(self.p_test.shape[0]):
            if self.p_test[i] == self.y_test[i]:
                count += 1
        self.acc = count / self.p_test.shape[0]
        print(f"Test accuracy: {self.acc}")
        return self.acc


    def confusion_matrix(self, loc='confusion_matrix.png'):
        p_labels = np.array(list(map(self.dataset.to_label, self.p_test)))
        y_labels = np.array(list(map(self.dataset.to_label, self.y_test)))
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_labels, p_labels)
        disp.figure_.suptitle(f"Confusion Matrix (acc: {self.acc})")
        plt.savefig(loc)


    def histogram_used_classes(self, loc='used_classes.png'):
        # Show histogram of used classes
        y_train_labels = np.array(list(map(self.dataset.to_label, self.y_train)))
        labels, counts = np.unique(y_train_labels[self.used], return_counts=True)
        ticks = range(len(counts))
        plt.bar(ticks, counts, align='center')
        plt.xticks(ticks, labels)
        plt.savefig(loc)
        plt.show()
    
    def accuracy_graph(self, used, accs, loc="acc-graph.png"):
        plt.suptitle("Accuracy graph (VGG16)")
        plt.plot([sum(x) for x in used], accs)
        plt.xlabel("Used samples")
        plt.ylabel("Accuracy")
        plt.savefig(loc)
        plt.show()


if __name__ == "__main__":
    Robot("tf-gpu-check")
    #Robot("full-learn")
    Robot('full-learn-acc-graph')
