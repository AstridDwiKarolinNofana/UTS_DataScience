import pandas as pd
import numpy as np
from numpy import genfromtxt

class class_dataset:

    X = []
    y = []
    name = ""

    def __init__(self, dataset_name=None):
        self.name = dataset_name
        self.pickDataset()

    def pickDataset(self): 
        if self.name == "heart_disease_patients":
            self.heart_disease_patients()
        elif self.name == "diabetes":
            self.diabetesDataset()
        elif self.name == "2D":
            self.artificialDataset()
        elif self.name == "glcm":
            self.glcmDataset()
        elif self.name == "dissimilarity":
            self.tenun_dissimilarity()

    def heart_disease_patients(self):
        print("DATASET Heart Disease Patients")
        df = genfromtxt('heart.csv', delimiter=',', skip_header=1)
        self.X = df[:, 0:13]
        y = df[:, 13:14]
        yy = []
        for i in range(len(y)):
            yy.append(y[i][0])
        self.y = yy

    def artificialDataset(self):
        print("DATASET 2D")
        file = pd.read_excel(open('dataset_artificial.xlsx', 'rb'))
        X = pd.DataFrame(file, columns=(['x', 'y']))
        self.X = np.array(X)
        y = pd.DataFrame(file, columns=(['T']))
        y = np.array(y)
        yy = []
        for i in range(len(y)):
            yy.append(y[i][0])
        self.y = yy

    def diabetesDataset(self):
        print("DATASET DIABETES")
        df = genfromtxt('diabetes_dataset.csv', delimiter=',', skip_header=1)
        # print(df.shape)
        self.X = df[:, 0:8]
        y = df[:, 8:9]
        yy = []
        for i in range(len(y)):
            # print(y[i][0])
            yy.append(y[i][0])
        self.y = yy

    def glcmDataset(self):
        print("DATASET glcm")
        df = genfromtxt('GLCM.csv.xls', delimiter=',', skip_header=1)
        # print(df.shape)
        self.X = df[:, 1:25]
        y = df[:, 25:26]
        yy = []
        for i in range(len(y)):
            # print(y[i][0])
            yy.append(y[i][0])
        self.y = yy

    def tenun_dissimilarity(self):
        print("DATASET Dissimilarity")
        df = genfromtxt('GLCMbaru.csv.xls', delimiter=',', skip_header=1)
        # print(df.shape)
        self.X = df[:, 1:5]
        y = df[:, 25:26]
        yy = []
        for i in range(len(y)):
            # print(y[i][0])
            yy.append(y[i][0])
        self.y = yy




