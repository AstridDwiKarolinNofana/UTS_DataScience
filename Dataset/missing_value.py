import numpy as np

class class_missing_value:

    X = []
    y = []
    nan = []
    name = ''

    def __init__(self,  data, missing_name=None):
        self.X = data.X
        self.y = data.y
        self.nan = np.isnan(data.X)
        self.name = missing_name
        self.pick()

    def pick(self):
        if self.name == "max":
            self.max()
        elif self.name == "min":
            self.min()

    def max(self):
        print('Mengatasi Missing Value Dengan Max')
        self.X[self.nan] = np.nanmax(self.X)

    def min(self):
        print('Mengatasi Missing Value Dengan min')
        self.X[self.nan] = np.nanmin(self.X)