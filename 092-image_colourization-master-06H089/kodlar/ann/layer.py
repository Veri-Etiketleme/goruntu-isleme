from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Abstract class for all ann layers
    """
    def initialize(self):
        pass

    def forward(self, _input):
        pass

    def back_prop(self, dl_dy, lr):
        pass
