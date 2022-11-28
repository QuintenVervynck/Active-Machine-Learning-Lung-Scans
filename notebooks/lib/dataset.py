import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
os.environ['KAGGLE_USERNAME'] = 'dumpstertrash'
os.environ['KAGGLE_KEY'] = 'fc29a78761d58a50c7b50d71f3f3a3f2'
from kaggle.api.kaggle_api_extended import KaggleApi


class Dataset():
    def __init__(self):
        self.path = "./dataset"
        self.subfolders = ["COVID/", "Normal/", "Viral Pneumonia/"]
        self.versions = ["images", "masks"]
        self.data = {}
        self.size = 0
        self.names = ["COVID", "Normal", "Viral Pneumonia"]
        self.revnames = {"COVID": 0, "Normal": 1, "Viral Pneumonia": 2}
        self.X = None
        self.y = None
        
        if not os.path.isdir(self.path):
            self.download()
        
        if not os.path.isfile(f"{self.path}/x.npy") or not os.path.isfile(f"{self.path}/y.npy"):
            self.load_local()
            self.restrict_size_per_class()
            self.prepare_images()
            self.save()
        else:
            self.load()
            
        print("Finished")
            
    def to_label(self, x:int):
        return self.names[x]
    
    
    def from_label(self, x:str):
        return self.revnames[x]
    
    def to_classlist(self, x:int):
        l = [0.0, 0.0, 0.0]
        l[x] = 1.0
        return l
    
    
    def from_classlist(self, x):
        return np.argmax(x)
    def download(self):                    
        if not os.path.isfile(f"{self.path}/covid19-radiography-database.zip"):
            print("Downloading...")
            api = KaggleApi()
            # env vars are set (see top of this file)
            api.authenticate()
            api.dataset_download_files('tawsifurrahman/covid19-radiography-database',
                                       path=self.path)
        if not os.path.isdir(f"{self.path}/COVID-19_Radiography_Dataset"):
            print("Unzipping...")
            with zipfile.ZipFile(f"{self.path}/covid19-radiography-database.zip","r") as zip_ref:
                zip_ref.extractall(f"{self.path}/")
        
    def load_local(self):
        print("Loading local files...")
        # Three main subfolders: COVID, Normal, Viral Pneumonia
        path = f"{self.path}/COVID-19_Radiography_Dataset/"
        # Every subfolder has images and their masks
        for subfolder in self.subfolders:
            self.data[subfolder + "images"] = [path + subfolder + "images/" + image for image in os.listdir(path + subfolder + "images")]
            self.data[subfolder + "masks"] = [path + subfolder + "masks/" + image for image in os.listdir(path + subfolder + "masks")]
            print(f"  Number of images in {subfolder}: {len(self.data[subfolder + 'images'])}")
            print(f"  Number of masks in {subfolder}: {len(self.data[subfolder + 'masks'])}")
            self.size += len(self.data[subfolder + 'images'])
        print(f"  Total size: {str(self.size)}")
    
    def show_unprocessed_examples(self):
        if "COVID/images" not in self.data:
            self.load_local()
        num_images = 3 # number of images to show
        examples_path = []
        examples_images = []
        for subfolder in self.subfolders:
            for version in self.versions:
                examples_path += self.data[subfolder + version][0:num_images]
                for image in examples_path[-num_images:]:
                    examples_images.append(plt.imread(image))
        _, axes = plt.subplots(nrows=len(self.subfolders) * len(self.versions), ncols=num_images, figsize=(10, 15))
        for ax, image, label in zip(axes.flatten(), examples_images, examples_path):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title("%s" % label.split("/")[-1])
    def restrict_size_per_class(self, size=1345):
        print("Restricting the number of images per class...")
        self.size = 3*size
        print("  Old sizes:")
        for subfolder in self.subfolders:
            print(f"   - {subfolder}: {len(self.data[subfolder + 'images'])}")
            self.data[subfolder + "images"] = self.data[subfolder + "images"][:size]
        print("  New sizes:")
        for subfolder in self.subfolders:
            print(f"   - {subfolder}: {len(self.data[subfolder + 'images'])}")
    
    
    def prepare_images(self):
        """
        Create the training data:
        - resize to 224x224
        - convert greyscale to RGB
        - apply the mask on the image
        """
        print("Preprocessing the images...")
        X = np.zeros((self.size, 224, 224, 3), dtype=np.uint8)
        index = 0
        test = 0
        for subfolder in self.subfolders:
            for i, image_path in enumerate(self.data[subfolder + "images"]):
                image = plt.imread(image_path)

                # Some images are not 8 bit, so we convert them
                if (len(image.shape) != 2):
                    img = Image.open(image_path)
                    new_img = img.convert("L")
                    new_img.save(image_path)

                # Convert image to 224,224,3
                img = Image.open(image_path)
                new_img = img.convert("RGB")
                new_img = new_img.resize(size=(224, 224))
                np_image = np.array(new_img)

                mask = self.data[subfolder + "masks"][i]
                # Convert mask to 224,224,3
                msk = Image.open(mask)
                new_msk = msk.resize(size=(224, 224))
                np_mask = np.array(new_msk)

                # Apply the mask
                new_image = np.multiply(np_image, np_mask)

                X[index] = new_image
                index += 1

        y = np.concatenate([np.array([subfolder[:-1]] * len(self.data[subfolder + "images"])) for subfolder in self.subfolders])
        y = np.array(list(map(self.from_label, y))) # map labels to ints of labels
        print(f"  size of training data: {len(X)}")
        print(f"  size of training data labels: {len(y)}")
        self.X = X
        self.y = y
        return (X,y)
    
    def show_processed_example(self):
        print(f"Example image with label {self.to_label(self.y[0])}:")
        plt.imshow(self.X[0])
    
              
    def save(self):
        print(f"Saving the preprocessed dataset...")
        np.save(f"{self.path}/x", self.X)
        np.save(f"{self.path}/y", self.y)
              
    def load(self):
        print(f"Loading local preprocessed dataset...")
        self.X = np.load(f"{self.path}/x.npy")
        self.y = np.load(f"{self.path}/y.npy")
    
    def split(self):
        # Split the dataset in training and test data
        # The data is found in self.X and self.y
        X_train, X_rem, y_train, y_rem = train_test_split(
            self.X, 
            self.y, 
            train_size=0.8, 
            shuffle=True,
            random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_rem,
            y_rem,
            train_size=0.5,
            shuffle=True,
            random_state=42)
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_valid & X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_valid & y_test shape: {y_test.shape}")
        return X_train, X_valid, X_test, y_train, y_valid, y_test