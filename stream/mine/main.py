import csv
import time
import pandas as pd

from stream.mine.MBStream import start

if __name__ == '__main__':
    dataset = "RBF3_40000(1000).csv"
    dataset_path = "withLabel"
    s = time.time()
    csv1 = pd.read_csv("../data/"+dataset_path+"/" + dataset, header=None)
    plot_evaluate_flag = True  # Whether to enable drawing function
    start(csv1, dataset, plot_evaluate_flag)



