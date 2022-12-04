import multiprocessing as mlp
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
os.environ['KAGGLE_USERNAME'] = 'dumpstertrash'
os.environ['KAGGLE_KEY'] = 'fc29a78761d58a50c7b50d71f3f3a3f2'
from kaggle.api.kaggle_api_extended import KaggleApi
from lib.xray.clahe import CLAHE


class Dataset():
    def __init__(self, path):
        self.path = path
        self.versions = ["images", "masks"]
        self.class_length = 1345
        self.data = {}
        self.size = 0
        self.names = ["COVID", "Normal", "Viral Pneumonia"]
        self.X = None
        self.y = None
        
    def load_local(self):
        self.X = np.load(f"{self.path}/x.npy")
        self.y = np.load(f"{self.path}/y.npy")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            train_size=0.85, 
            shuffle=True,
            random_state=42
        )
        self.size = self.X_train.shape[0]
        
    def split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def to_label(self, x:int):
        return self.names[x]
    
    def from_label(self, x:str):
        return self.names.index()
    
    def to_classlist(self, x:int):
        l = [0.0, 0.0, 0.0]
        l[x] = 1.0
        return l
    
    def from_classlist(self, x):
        return np.argmax(x)

    def load_images(self):
        # Three main subfolders: COVID, Normal, Viral Pneumonia
        # Every subfolder has images and their masks
        for name in self.names:
            self.data[name] = {}
            img_path = f"{self.path}/{name}/images"
            mask_path = f"{self.path}/{name}/masks"
            self.data[name]["images"] = [f"{img_path}/{image}" for image in os.listdir(img_path)]
            self.data[name]["masks"] = [f"{mask_path}/{mask}" for mask in os.listdir(mask_path)]            

    def show_image(self, i):
        print(f"Example image with label {self.to_label(self.y[i])}:")
        plt.imshow(self.X[i])

def initialize_dataset():
    names = ["COVID", "Normal", "Viral Pneumonia"]
    """
    print("Downloading...")
    api = KaggleApi()
    # env vars are set (see top of this file)
    api.authenticate()
    api.dataset_download_files('tawsifurrahman/covid19-radiography-database', path="./dataset")
    print("Unzipping...")
    with zipfile.ZipFile(f"./dataset/covid19-radiography-database.zip","r") as zip_ref:
        zip_ref.extractall(f"./dataset/")
    os.remove("./dataset/covid19-radiography-database.zip")
        
    print(f"Setting up the directory structure...")
    for name in names:
        os.mkdir(f"dataset/{name}")
        for version in ["images", "masks"]:
            os.mkdir(f"dataset/{name}/{version}")
            file_list = [f"{file}" for file in os.listdir(f"./dataset/COVID-19_Radiography_Dataset/{name}/{version}")]
            file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            file_list = file_list[:1345] # take first 1345 files
            
            for file in file_list:
                try:
                    os.rename(f"./dataset/COVID-19_Radiography_Dataset/{name}/{version}/{file}", f"./dataset/{name}/{version}/{file}")
                except:
                    print("Error while moving file : ", file)
    print("Removing the old database")
    shutil.rmtree("./dataset/COVID-19_Radiography_Dataset")
    
    print(f"Converting size of the images and masks to 244x244x3")
    for name in names:
        for file in os.listdir(f"./dataset/{name}/images"):
            img_path = f"./dataset/{name}/images/{file}"
            mask_path = f"./dataset/{name}/masks/{file}"

            image = plt.imread(img_path)

            # Some images are not 8 bit, so we convert them
            if (len(image.shape) != 2):
                img = Image.open(img_path)
                new_img = img.convert("L")
                new_img.save(img_path)

            # Convert image to 224,224,3
            img = Image.open(f"./dataset/{name}/images/{file}")
            new_img = img.convert("RGB")
            new_img = new_img.resize(size=(224, 224))
            new_img.save(img_path)

            # Convert mask to 224,224,3
            msk = Image.open(mask_path)
            new_msk = msk.resize(size=(224, 224))
            new_msk.save(mask_path)
"""
    print("Applying the masks on the images...")
    X = np.zeros((1345*3, 224, 224, 3), dtype=np.float32)
    index = 0
    for name in names:
        for file in os.listdir(f"./dataset/{name}/images"):
            img = Image.open(f"./dataset/{name}/images/{file}") # Reads in uint8

            # change grey to rgb
            if img.mode == "L":
                img = img.convert("RGB")
            
            new_img = np.array(img).astype(np.float32)
            new_img /= 255
                
            np_image = np.array(new_img)
            np_mask = plt.imread(f"./dataset/{name}/masks/{file}") # Reads in float32
            # Apply the mask
            new_image = np.multiply(np_image, np_mask)
            # save to array
            X[index] = new_image
            index += 1
    print("Gathering the labels...")
    y = np.concatenate([np.array([name] * len(os.listdir(f"./dataset/{name}/images"))) for name in names])
    def from_label(x:str):
        return names.index(x)
    y = np.array(list(map(from_label, y))) # map labels to ints of labels

    print(f"Saving the preprocessed dataset...")
    np.save(f"./dataset/x.npy", X)
    np.save(f"./dataset/y.npy", y)
    

def enhance(files, name):
    for file in files:
        CLAHE(f"./dataset/{name}/images/{file}", f"./enhanced/{name}/images", 100, 150, 1).run()
    
def enhance_dataset():
    names = ["COVID", "Normal", "Viral Pneumonia"]
    """
    processes = []
    cores = (mlp.cpu_count() - 2) // 3
    print(f"Using {cores * 3} cores to enhance images")
    
    os.mkdir("./enhanced")

    for name in names:
        print(f"Enhancing: {name}")
        os.mkdir(f"./enhanced/{name}")
        os.mkdir(f"./enhanced/{name}/images")
        shutil.copytree(f"./dataset/{name}/masks", f"./enhanced/{name}/masks")
        for files in np.array_split(os.listdir(f"./dataset/{name}/images"), cores):
            p = mlp.Process(target=enhance, args=(files, name))
            p.start()
            processes.append(p)

    print(f"Made a total of {len(processes)} processes to enhance images")
    for process in processes:
        process.join()
        
    print(f"Converting size of the images and masks to 244x244x3")
    for name in names:
        for file in os.listdir(f"./enhanced/{name}/images"):
            img_path = f"./enhanced/{name}/images/{file}"
            mask_path = f"./enhanced/{name}/masks/{file}"

            image = plt.imread(img_path)

            # Some images are not 8 bit, so we convert them
            if (len(image.shape) != 2):
                img = Image.open(img_path)
                new_img = img.convert("L")
                new_img.save(img_path)
                
            # Convert image to 224,224,3
            img = Image.open(img_path)
            new_img = img.convert("RGB")
            new_img = new_img.resize(size=(224, 224))
            new_img.save(img_path)
            
            # Convert mask to 224,224,3
            msk = Image.open(mask_path)
            new_msk = msk.resize(size=(224, 224))
            new_msk.save(mask_path)
"""
    print("Applying the masks on the images...")
    X = np.zeros((1345*3, 224, 224, 3), dtype=np.float32)
    index = 0
    for name in names:
        for file in os.listdir(f"./enhanced/{name}/images"):
            img = Image.open(f"./enhanced/{name}/images/{file}")

            # change grey to rgb
            if img.mode == "L":
                img = img.convert("RGB")
            
            new_img = np.array(img).astype(np.float32)
            new_img /= 255
                
            np_image = np.array(new_img)
            np_mask = plt.imread(f"./enhanced/{name}/masks/{file}") # Reads in float32
            # Apply the mask
            new_image = np.multiply(np_image, np_mask)
            # save to array
            X[index] = new_image
            index += 1
    print("Gathering the labels...")
    y = np.concatenate([np.array([name] * len(os.listdir(f"./enhanced/{name}/images"))) for name in names])
    def from_label(x:str):
        return names.index(x)
    y = np.array(list(map(from_label, y))) # map labels to ints of labels
    
    print(f"Saving the preprocessed dataset...")
    np.save(f"./enhanced/x.npy", X)
    np.save(f"./enhanced/y.npy", y)