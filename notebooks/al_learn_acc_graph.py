from matplotlib import pyplot as plt
import numpy as np

from lib.dataset import Dataset
from lib.model import Model

if __name__ == "__main__":
    mode = "calculate"  # load or calculate

    if mode == "calculate":
        print("AL LEARN ACC GRAPH")
        dataset = Dataset()

        def uncertainty(model, X_pool, query_size):
            query_size = min(query_size, X_pool.shape[0])
            # get predictions on remaining training data
            p_pool = model.predict(X_pool)
            # get indexes of most uncertain predictions) (if the max is low, then the prediction is uncertain)
            idxs = np.max(p_pool, axis=1).argsort()[:query_size]
            return idxs

        steps = [i for i in range(50, 1050, 50)] + [i for i in range(1100, 3201, 100)]
        models = []

        for i, max_imgs in enumerate(steps):
            print(f"TRAINING MODEL {i+1}/{len(steps)} ON {max_imgs} IMGS")
            model = Model("vgg16", dataset)
            models.append(model)
            
            model.train_al(max_imgs=max_imgs, strategy=uncertainty).test().clean_full()

            with open("results/al_learn/al_learn_acc_graph_checkpoints.txt", "a+") as f:
                f.write(f"{str(models[-1])}\n")
            plt.suptitle("Accuracy graph (VGG16)")
            plt.xlabel("Used samples")
            plt.ylabel("Accuracy")
            plt.plot([sum(m.used) for m in models], [m.acc for m in models])
            plt.savefig("results/al_lear/al_learn_acc_graph.png")
    
    elif mode == "load":
        datafile = "results/al_learn/al_learn_acc_graph_checkpoints.txt"
        used = []
        accs = []
        with open (datafile, "r") as f:
            while(line:=f.readline()):
                line = line.split()
                used.append(int(line[3]))
                accs.append(float(line[-1][:-1]))        
        plt.suptitle("Accuracy graph (VGG16)")
        plt.xlabel("Used samples")
        plt.ylabel("Accuracy")
        plt.plot(used, accs)
        plt.show()