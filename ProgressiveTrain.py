from MLP import Network
import os
from Trainer import Trainer
import csv
import numpy as np
from numpy import float32


def readMyFile(filename):
    data = []
    labels = []
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            n = row.__len__()
            data.append([row[index] for index in range(0,n-1)])
            labels.append(row[n-1])
    print(data)
    print(labels)
    return np.asarray(data, dtype=float32), np.asarray(labels,  dtype=float32)



# # Ricordare di cambiare i path dentro la radice che sceglierete dovrete creare la cartella config con dentro i json, consiglio di lasciare tutto come lo fornisco io
ROOT_DIR = os.path.abspath(os.curdir)
os.chdir(ROOT_DIR)
jsonStructureDir = ROOT_DIR + "\config\structure.txt"
jsonTrainDir = ROOT_DIR + "\config\\train.txt"
jsonProgressiveTrainDir = ROOT_DIR + "\config\\progressiveTrain.txt"


#7 esempio di test classe Trainer stand-alone con allenamento progressivo e riutilizzo rete
trainer_5 = Trainer(jsonTrainDir, jsonProgressiveTrainDir)
data,labels = readMyFile(trainer_5.path)
net_test_5 = trainer_5.createNet(data, labels, jsonStructureDir)
trainer_5.progressiveTrain()
net_test_5.saveNetwork()

# #8 Uso della rete salvata, per usare una rete salvata basta mettere il suo nome nel file jsonStructureDir e istanzianziarla con rebuild
# netTest_6=Network(jsonStructureDir,rebuild=True)
# predProb = netTest_6.predict(xTest)
# yHat = np.where(predProb < 0.5, 0, 1)
# acc, trueVector, oneHotYTest = netTest_6.acc(yHat,yTest)
# print("Test Accuracy %.2f" % acc)
