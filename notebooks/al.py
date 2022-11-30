from lib.dataset import Dataset
# the dataset
dataset = Dataset()


# the query strategies to try

from lib.queries import random, uncertainty, margin, entropy
qss = [("random", random), ("uncertainty", uncertainty), ("margin", margin), ("entropy", entropy)]


# the amount of images the query strategy sould return in each iteration

query_size = 25


# create the files for the results for each query strategy

import os
for qs_name, _ in qss:
    if not os.path.exists(f"results/al_learn/{qs_name}"):
        os.mkdir(f"results/al_learn/{qs_name}")
    if not os.path.exists(f"results/al_learn/{qs_name}/checkpoints.txt"):
        with open(f"results/al_learn/{qs_name}/checkpoints.txt", "w") as f:
            f.write("")
            

# we now train models for every query strategy, and save accuracy data along the way

import time 
from lib.activelearner import ActiveLearner
# variables for colored output
OKCYAN = '\033[96m'
ENDC = '\033[0m'

query_size = 32
max_train_imgs = 512  # dataset has 3429 in training set
epochs = 8
batch_size = 16

for i, (name, qs) in enumerate(qss):
    time_start = time.time()
    
    accs = []
    
    active_learner = ActiveLearner(dataset, architecture="vgg16").with_params(epochs, batch_size)
    
    # the steps we will take (steps over the size of the data)
    steps = [i for i in range(query_size, max_train_imgs+1, query_size)]
    
    for j, max_imgs in enumerate(steps):
        time_round_start = time.time()
        print(f"{OKCYAN}<{name} ({i+1}/{len(qss)}) | using {max_imgs} images ({j+1}/{len(steps)})>{ENDC}")
        
        # select less samples if we can't select the full query_size anymore
        n = min([query_size, (max_imgs - sum(active_learner.used))])

        # select samples
        active_learner.select_samples(qs, n)

        # train the model
        active_learner.train()

        # test the model
        active_learner.test()

        # save metrics to file
        with open(f"results/al_learn/{name}/checkpoints.txt", "a") as f:
            f.write(f"{str(active_learner)}\n")

        time_round = int(time.time() - time_round_start) // 60
        time_total = int(time.time() - time_start) // 60

        print(f"{OKCYAN}----> finished round in {time_round} mins  (total run time for {name}: {time_total} mins){ENDC}")

