def download():
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi
    import zipfile

    os.environ['KAGGLE_USERNAME'] = 'dumpstertrash'
    os.environ['KAGGLE_KEY'] = 'fc29a78761d58a50c7b50d71f3f3a3f2'
    
    api = KaggleApi()
    api.authenticate()
    print("Downloading...")
    api.dataset_download_files('tawsifurrahman/covid19-radiography-database',
                               path="./ActiveLearning_ImageClassification")

    print("Unzipping...")
    if os.path.isfile("./ActiveLearning_ImageClassification/covid19-radiography-database.zip") \
        and not os.path.isdir("./ActiveLearning_ImageClassification/COVID-19_Radiography_Dataset"):
        with zipfile.ZipFile("./ActiveLearning_ImageClassification/covid19-radiography-database.zip","r") as zip_ref:
            zip_ref.extractall("./ActiveLearning_ImageClassification/")
    print("Finished")


def test():
    print("ok")