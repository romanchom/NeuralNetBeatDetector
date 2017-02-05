from DataBase import DataBase
from NeuralNet import NeuralNet
from threading import Thread

from kivy.app import App
from GraphViewerWidget import GraphViewerWidget
import numpy as np

class Beat(App):
    def __init__(self, **kwargs):
        super(Beat, self).__init__(**kwargs)
        self.base = DataBase()
        self.train_thread = Thread(target=self.train_nn)
        self.should_exit = False

    def build(self):
        self.graph = GraphViewerWidget()
        self.graph.set_graph("truth", [0, 0, 1, 1], (0, 1, 0))
        self.graph.set_graph("predictions", [0, 1, 1, 0], (1, 0, 0))
        self.train_thread.start()
        return self.graph

    def on_stop(self):
        self.should_exit = True
        self.train_thread.join()

    def train_nn(self):
        self.nn = NeuralNet()
        self.nn.load()
        self.base.load_bin("C:\\Users\\Romek\\Desktop\\maniac_examples\\", count=-1)
        for i in range(10000):
            if(self.should_exit): break
            batch_size = 50
            examples, labels = self.base.get_batch(batch_size)
            predictions, ground_truth = self.nn.train(examples, labels, batch_size, 10)
            print(i)
            
            xs = np.linspace(0.0, 1.0, len(predictions))
            
            interleaved = np.zeros(len(predictions) * 2)
            interleaved[0::2] = xs
            interleaved[1::2] = ground_truth
            self.graph.set_graph("truth", interleaved.tolist(), (0, 1, 0))
            
            interleaved = np.zeros(len(predictions) * 2)
            interleaved[0::2] = xs
            interleaved[1::2] = predictions
            self.graph.set_graph("predictions", interleaved.tolist(), (1, 0, 0))
            if(i % 100 == 0):
                self.nn.save()
                print("save")

        self.nn.save()
        print("train done")

if __name__ == '__main__':
    Beat().run()