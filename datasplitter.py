import pandas as pd
import numpy as nu
import random
def main():
    file = open("./data_files/train.csv", "r")
    filee = open("train.csv", "w", newline = '')
    fileee = open("test.csv", "w", newline = '')
    data = pd.read_csv(file, sep=",")
    data = data.sample(frac=1).reset_index(drop=True)
    test = data[0:int(len(data)/4)]
    data = data[int(len(data)/4): len(data)]
    data.to_csv(filee, sep = ',', index = False)
    test.to_csv(fileee, sep = ',', index = False)

if __name__ == "__main__":
    main()