import numpy as np
import os
import multiprocessing as mlp


from lib.dataset import Dataset

# dataset = Dataset()


if __name__ == '__main__':
    dataset = Dataset(enhance_data=True)
    processes = []
    cores = (mlp.cpu_count() - 2) // 3
    print(f"Using {cores * 3} cores to enhance image")

    for subfolder in dataset.subfolders:
        print(f"Enhancing: {dataset.enhanced}/COVID-19_Radiography_Dataset/{subfolder}images")
        os.makedirs(f"{dataset.enhanced}/COVID-19_Radiography_Dataset/{subfolder}images", exist_ok=True)

        for files in np.array_split(dataset.data[subfolder + "images"], cores):
            p = mlp.Process(target=dataset.enhance, args=(files, subfolder))
            p.start()
            processes.append(p)
    
    print(f"Made a total of {len(processes)} to enhance images")
    for process in processes:
        process.join()


