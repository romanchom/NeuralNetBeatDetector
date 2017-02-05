import os
import csv
import numpy as np

class DataBase:

    def __init__(self):
        self.examples = []
        self.labels = []

    def load_csv(self, directory, count=-1):
        framesPerExample = 200
        exampleSize = 120
        x = 0
        for file in os.listdir(directory):
            if(not file.endswith(".csv")): continue
            if(x == count): break
            x += 1
            print(file)
            with open(directory + file, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                labels = np.zeros((framesPerExample, 2))
                examples = np.zeros((framesPerExample, exampleSize))
                i = 0
                for row in reader:
                    isBeat = int(row[0])
                    # one hot encoding
                    labels[i, isBeat] = 1
                    for j in range(exampleSize):
                        examples[i, j] = float(row[j + 1])
                    i += 1
                self.labels.append(labels)
                self.examples.append(examples)

    def load_bin(self, directory, count=-1):
        framesPerExample = 200
        exampleSize = 120
        x = 0
        for file in os.listdir(directory):
            if(not file.endswith(".bin")): continue
            if(x == count): break
            x += 1
            print(file)
            data = np.fromfile(directory + file, dtype=np.float32)
            data = np.reshape(data, (framesPerExample, exampleSize + 2))
            self.labels.append(data[:, 0:2])
            self.examples.append(data[:, 2:122])

    def get_batch(self, size):
        indices = np.random.choice(np.arange(len(self.labels)), size)
        examples = [self.examples[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        return examples, labels
