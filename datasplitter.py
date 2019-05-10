import pandas as pd

def main():
    file = open("./data_files/adult.data.txt", "r")
    filee = open("adult.data.1-4th.txt", "w", newline = '')
    fileee = open("adult.test.1-4th.txt", "w", newline = '')
    data = pd.read_csv(file, sep=",")
    data = data.sample(frac=1).reset_index(drop=True)
    test = data[0:int(len(data)/4)]
    data = data[int(len(data)/4): len(data)]
    data.to_csv(filee, sep = ',', index = False)
    test.to_csv(fileee, sep = ',', index = False)

if __name__ == "__main__":
    main()
