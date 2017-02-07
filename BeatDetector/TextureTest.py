from kivy.app import App
from GraphViewerWidget import GraphViewerWidget
import numpy as np

class MyApp(App):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        
    def build(self):
        self.graph = GraphViewerWidget()
        self.graph.set_graph("truth", [0, 0, 1, 1], (0, 1, 0))
        self.graph.set_graph("prediction", [0, 1, 1, 0], (1, 0, 0))
        return self.graph


if __name__ == '__main__':
    MyApp().run()