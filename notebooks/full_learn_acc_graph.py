import matplotlib.pyplot as plt

from lib.dataset import Dataset
from lib.model import Model

if __name__ == "__main__":    
    print("FULL LEARN ACC GRAPH")
    dataset = Dataset()
        
    steps = [i for i in range(50, 1050, 50)] + [i for i in range(1100, 3201, 100)]
    models = []

    for i, max_imgs in enumerate(steps):
        print(f"TRAINING MODEL {i+1}/{len(steps)} ON {max_imgs} IMGS")
        models.append(
            Model("vgg16", dataset).limit_training_data(max_imgs).train().test().clean_full()
        )
        with open("full_learn_acc_graph_checkpoints.txt", "a+") as f:
            f.write(f"{str(models[-1])}\n")
        plt.suptitle("Accuracy graph (VGG16)")
        plt.xlabel("Used samples")
        plt.ylabel("Accuracy")
        plt.plot([sum(m.used) for m in models], [m.acc for m in models])
        plt.savefig("full_learn_acc_graph.png")

    