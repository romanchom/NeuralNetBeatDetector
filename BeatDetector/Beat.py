from DataBase import DataBase
from NeuralNet import NeuralNet
from threading import Thread

from kivy.app import App
from GraphViewerWidget import GraphViewerWidget
import numpy as np
from kivy.clock import Clock

class Beat(App):
    def __init__(self, **kwargs):
        super(Beat, self).__init__(**kwargs)
        self.training_set = DataBase()
        self.validation_set = DataBase()
        self.train_thread = Thread(target=self.train_nn)
        self.should_exit = False
        self.should_load = True
        self.should_save = True

    def build(self):
        self.graph = GraphViewerWidget()
        self.graph.set_graph("truth", [0, 0, 1, 1], (0, 1, 0))
        self.graph.set_graph("prediction", [0, 1, 1, 0], (1, 0, 0))
        self.train_thread.start()
        return self.graph

    def on_stop(self):
        self.should_exit = True
        self.train_thread.join()

    def train_nn(self):
        batch_size = 1022
        no_improvement_limit = 20

        self.nn = NeuralNet()
        if self.should_load: self.nn.load()
        self.training_set.load_bin("C:\\BeatDetectorData\\TrainingBin\\", count=-1)
        self.validation_set.load_bin("C:\\BeatDetectorData\\ValidationBin\\", count=-1)
        
        valid_ex, valid_lab = self.training_set.get_batch(batch_size)

        
        last_improvement = 0
        best_cross_entropy = float('inf')
        for i in range(10000):
            
            # TRAIN ON ENTIRE DATA SET IN RANDOM ORDER
            epoch = self.training_set.get_epoch(batch_size)
            for examples, labels in epoch:
                if(self.should_exit): break
                self.nn.train(examples, labels)
                print('.', end='', flush=True)
            
            if(self.should_exit): break


            print("")
            # VALIDATE ON RANDOM SUBSET
            cross_entropy = self.nn.validate(valid_ex, valid_lab)
            print("Epoch: {}, Cross_entropy: {}".format(i, cross_entropy))
            
            if(cross_entropy < best_cross_entropy):
                last_improvement = i
                best_cross_entropy = cross_entropy
            else:
                print("WARNING, no improvement")

            # EXAMINE ONE EXAMPLE FROM VALIDATION SET TO DRAW PRETTY GRAPHS
            example, ground_truth = self.training_set.get_any() 
            prediction = self.nn.examine(example)
            #prediciton_new = self.nn.single_step_examine(example)
            
            self.gui_ex = example


            ground_truth = ground_truth[:, 1]
            prediction = prediction[:, 1]
            xs = np.linspace(0.0, 1.0, len(prediction))
            
            interleaved = np.zeros(len(prediction) * 2)
            interleaved[0::2] = xs
            interleaved[1::2] = ground_truth
            self.gui_truth = interleaved.tolist()

            interleaved[1::2] = prediction
            self.gui_prediction = interleaved.tolist()

            #interleaved[1::2] = prediciton_new
            #self.gui_prediction_new = interleaved.tolist()

            if self.should_save: self.nn.save()
            Clock.schedule_once(self.update_gui)

            if(i - last_improvement >= no_improvement_limit):
                print("No improvement for last {}, aborting", no_improvement_limit)
                break

        print("train done")
        self.nn.export_to_protobuffer("./export")
        

    def update_gui(self, dt):
        self.graph.set_spectrogram(self.gui_ex)
        self.graph.set_graph("truth", self.gui_truth, (0, 1, 0))
        self.graph.set_graph("prediction", self.gui_prediction, (1, 0, 0))
        #self.graph.set_graph("prediction_new", self.gui_prediction_new, (0, 0, 1))


if __name__ == '__main__':
    app = Beat()
    app.should_load = True
    app.should_save = True
    app.run()