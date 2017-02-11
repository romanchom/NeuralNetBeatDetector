from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.core.window import Window
from kivy.graphics.texture import Texture

import numpy as np
from Config import Config

class GraphViewerWidget(Widget):
    def __init__(self, **kwargs):
        super(GraphViewerWidget, self).__init__(**kwargs)
        Window.size = (1600, 900)
        self.texture = Texture.create(size=(Config.framesPerExample, Config.exampleLength))
        self.buffer = np.random.random(Config.framesPerExample * Config.exampleLength).astype('float32')
        self.texture.blit_buffer(self.buffer, colorfmt='luminance', bufferfmt='float')
        self.graphs = {}
        with self.canvas:
            self.image = Rectangle(texture=self.texture, size=Window.size)

    def set_spectrogram(self, data):
        flat = np.zeros(data.shape, dtype='float32')
        flat = np.transpose(data, [1, 0])
        flat = np.reshape(flat, [-1])

        for i in range(len(flat)):
            self.buffer[i] = float(flat[i] / 12)
        self.texture.blit_buffer(self.buffer, colorfmt='luminance', bufferfmt='float')
        self.image.texture=self.texture


    def set_graph(self, name, points, color=(1, 0, 0)):
        size = Window.size
        for i in range(0, len(points), 2):
            points[i] *= size[0]
            points[i + 1] *= size[1]
        
        if(name in self.graphs):
            line = self.graphs[name]
            line.points = points
        else:
            with self.canvas:
                Color(*color)
                line = Line(points = points)
                self.graphs[name] = line
