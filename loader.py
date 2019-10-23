# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:19:38 2019

@author: giova
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


# import vectorized as vect
# from view import View

class Loader(object):

    def __init__(self, filename, path='', labelCol=-1, pos_class=None):
        self.filename, self.path = filename, path
        self.classes, self.features = None, None
        self.X, self.y = None, None
        self.load(labelCol=labelCol)

    def load(self, labelCol=-1, header=True, pos_class=None):
        with open(self.path + self.filename) as f:
            content = f.readlines()[1:]
        self.make_header(content, labelCol=labelCol, pos_class=pos_class, header=header)
        self.make_dataset(content, labelCol=labelCol, pos_class=pos_class)
        self.scale()

    def make_dataset(self, content, labelCol=-1, pos_class=-1):
        X, y = self.parse_data(content, labelCol=labelCol)
        self.classes = list(set(y))
        self.X = self.encode_data(X)
        self.y = self.encode_labels(y, pos_class=pos_class)

    def make_header(self, content, labelCol=-1, pos_class=-1, header=True):
        if header:
            self.features = [feature.strip() for feature in content[0].split(',')]
            content.pop(0)
        else:
            self.features = ["F" + str(k + 1) for k, item in enumerate(content[0])]

    def scale(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.X = binarize(scaler.fit_transform(self.X)).astype(int)
        self.X = scaler.fit_transform(self.X)

    def as_data_labels(self):
        return self.X, self.y

    def encode_data(self, data):
        return np.array(data).astype(float)

    def encode_labels(self, labels, pos_class=None):
        if not pos_class:
            pos_class = self.classes[0]
        elif type(pos_class) == int:
            pos_class = self.classes[pos_class]
        return (np.array(labels) == pos_class).astype(int)

    def parse_data(self, lines, labelCol=-1):
        X, y = list(), list()
        for line in lines:
            split = [t.strip() for t in line.split(",")]
            if labelCol == 0:
                y.append((split[0],));  X.append(split[1:])
            else:
                y.append(split[-1]); X.append(split[:-1])
        return X, y


if __name__ == '__main__':
    pass
