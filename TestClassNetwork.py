import numpy as np
from sklearn.datasets import make_moons
from Network import Network
import os
from Trainer import Trainer
## prendo un dataset di test dalla repository di sklearn

np.random.seed(0)
X, Y = make_moons(5000, noise= 0.2)
## Ricordare di cambiare i path
ROOT_DIR = os.path.abspath(os.curdir)
jsonStructureDir = ROOT_DIR + "\structure.txt"
jsonTrainDir = ROOT_DIR + "\\train.txt"
jsonProgressiveTrainDir = ROOT_DIR + "\\train.txt"

# esempio di test: classe Network stand-alone effettua solo un training di tutta la reta secondo le informazioni contenute nel json "structure.txt"

netTest_1 = Network(jsonStructureDir, XStandAlone= X, YStandAlone = Y)
Network.train(netTest_1)
predProb = netTest_1.predict()
yHat = np.where(predProb < 0.5, 0, 1)
acc = netTest_1.acc(yHat)
print("Test Accuracy %.2f" % acc)


# esempio di test classe Trainer + Network (simple train) (file "structure.txt"

trainer = Trainer(jsonTrainDir)
xTrain, xTest, yTrain, yTest = trainer.split(X = X, Y = Y)
netTest_2 = Network(jsonStructureDir, x=xTrain, y=yTrain)
trainer.simpleTrain(netTest_2)
netTest_2.saveNetwork(ROOT_DIR + "\\netTest_2")

#
