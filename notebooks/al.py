import numpy as np
from matplotlib import pyplot as plt
import os
import time 

from lib.dataset import Dataset
from lib.model import Model


def random(model, X_pool, query_size):
    query_size = min(query_size, X_pool.shape[0])
    # select random indexes
    idxs = np.random.choice(X_pool.shape[0], query_size, replace=False)
    return idxs


def uncertainty(model, X_pool, query_size):
            query_size = min(query_size, X_pool.shape[0])
            # get predictions on remaining training data
            p_pool = model.predict(X_pool, verbose=0)
            # get indexes of most uncertain predictions) (if the max is low, then the prediction is uncertain)
            idxs = np.max(p_pool, axis=1).argsort()[:query_size]
            return idxs

qss = [("random", random), ("uncertainty", uncertainty)]


# create the files for the results
for qs_name, _ in qss:
    # check if the directory qs_name exists
    if not os.path.exists(f"results/al_learn/{qs_name}"):
        os.mkdir(f"results/al_learn/{qs_name}")
    if not os.path.exists(f"results/al_learn/{qs_name}/checkpoints.txt"):
        with open(f"results/al_learn/{qs_name}/checkpoints.txt", "w") as f:
            f.write("")



dataset = Dataset()


query_size = 25
steps = [i for i in range(query_size, 3200, query_size)]

OKCYAN = '\033[96m'
ENDC = '\033[0m'


for i, (name, qs) in enumerate(qss):
    time_start = time.time()
    accs = []
    model = Model("vgg16", dataset).set_epochs(10).set_batch_size(16)
    X_pool = model.X_train
    y_pool = model.y_train
    model.X_train = None
    model.y_train = None
    for j, max_imgs in enumerate(steps):
        time_round_start = time.time()
        print(f"{OKCYAN}<{name} ({i+1}/{len(qss)}) | using {max_imgs} images ({j+1}/{len(steps)})>{ENDC}")
        imgs = np.min([query_size, (max_imgs - sum(model.used))])
        print(f"model: using query strategy to select {imgs} images")
        idxs = qs(model.model, X_pool, imgs)
        # move the selected images to the training set
        model.X_train = X_pool[idxs]
        model.y_train = y_pool[idxs]
        # remove the selected images from the pool
        X_pool = np.delete(X_pool, idxs, axis=0)
        y_pool = np.delete(y_pool, idxs, axis=0)
        # update size
        model.size = len(idxs)
        # train the model
        model.train()
        # test the model
        model.test()
        
        # save metrics
        accs.append(model.acc)

        with open(f"results/al_learn/{name}/checkpoints.txt", "a") as f:
            f.write(f"{str(model)}\n")
        plt.suptitle(f"Accuracy graph (VGG16 - {name})")
        plt.xlabel("Used samples")
        plt.ylabel("Accuracy")
        plt.plot(steps[:j+1], accs)
        plt.savefig(f"results/al_learn/{name}/acc_graph.png")
        plt.clf()
        
        time_round = int(time.time() - time_round_start) // 60
        time_total = int(time.time() - time_start) // 60
        
        print(f"{OKCYAN}----> finished round in {time_round} mins  (total run time for {name}: {time_total} mins){ENDC}")
    