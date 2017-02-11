import os
import csv
import numpy as np
from random import shuffle
from random import randint
from Config import Config

class DataBase:

    def __init__(self):
        self.examples = []
        self.labels = []

    def load_csv(self, directory, count=-1):
        x = 0
        for file in os.listdir(directory):
            if(not file.endswith(".csv")): continue
            if(x == count): break
            x += 1
            #print(file)
            with open(directory + file, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                labels = np.zeros((Config.framesPerExample, 2))
                examples = np.zeros((Config.framesPerExample, exampleLength))
                i = 0
                for row in reader:
                    isBeat = int(row[0])
                    # one hot encoding
                    labels[i, isBeat] = 1
                    for j in range(exampleLength):
                        examples[i, j] = float(row[j + 1])
                    i += 1
                self.labels.append(labels)
                self.examples.append(examples)

    def load_bin(self, directory, count=-1):
        exampleLength = 120
        x = 0
        for file in os.listdir(directory):
            if(not file.endswith(".bin")): continue
            if(x == count): break
            x += 1
            #print(file)
            data = np.fromfile(directory + file, dtype=np.float32)
            data = np.reshape(data, (Config.framesPerExample, exampleLength + 2))
            self.labels.append(data[:, 0:2])
            self.examples.append(data[:, 2:122])

    def get_batch(self, size):
        indices = np.random.choice(np.arange(len(self.labels)), size)
        examples = [self.examples[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        return examples, labels

    def get_epoch(self, size):
        count = len(self.examples)
        indices = np.arange(count)
        shuffle(indices)
        ex = [self.examples[i] for i in indices]
        lb = [self.labels[i] for i in indices]
        exs = [ex[i:i+size] for i in range(0, count - size + 1 , size)]
        lbs = [lb[i:i+size] for i in range(0, count - size + 1 , size)]
        return zip(exs, lbs)

    def get_any(self):
        index = randint(0, len(self.examples) - 1)
        return self.examples[index], self.labels[index]

