from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.graphics import Color, Line, Ellipse
from kivy.core.window import Window


class GraphViewerWidget(Widget):
    def __init__(self, **kwargs):
        super(GraphViewerWidget, self).__init__(**kwargs)
        self.graphs = {}

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
