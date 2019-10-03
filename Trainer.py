import json
from sklearn.model_selection import train_test_split
from Network import Network
import numpy as np

class Trainer(object):

    def __init__(self, jsonTrainDir= '', jsonTrainProgDir = ''):
        self.getJsonData(jsonTrainDir)


    def getJsonData(self,jsonTrainDir, jsonTrainProgDir = ''):
        self.trainPlane = []
        if ( jsonTrainProgDir !='' ):
            with open(jsonTrainDir, "r") as file:
                netData = file.read().replace("\n", "")
            netDataDict = json.loads(netData)
            self.trainPlane =  netDataDict["train_plane"]
            pass
        else:
            with open(jsonTrainDir, "r") as file:
                netData = file.read().replace("\n", "")
            netDataDict = json.loads(netData)
            self.nIters = netDataDict["n_iters"]
        self.networkName = netDataDict["network_name"]
        self.testSize = netDataDict["test_size"]
        self.randomState = netDataDict["random_state"]
        self.learningRate = netDataDict["learning_rate"]


    def split(self, X, Y):
        xTrain, self.xTest, yTrain, self.yTest = train_test_split(X, Y, test_size=self.testSize, random_state=self.randomState)
        return xTrain, self.xTest, yTrain, self.yTest


    def simpleTrain(self, net, nConvergence=0):
        if(nConvergence == 0):
            nConvergence = self.nIters
        Network.train(net, nIters= self.nIters, nConvergence= nConvergence)
        predProb = net.predict(self.xTest)
        yHat = np.where(predProb < 0.5, 0, 1)
        acc = net.acc(yHat, net.label_encoding(self.yTest))
        print("Test Accuracy %.2f" % acc)


    def progressiveTrain(self,net):
        for train in self.trainPlane:
            net.unfreezTrainableVar();
            for layer in train["freez_layers"]:
                net.freezTrainableVar(layer)
            Network.train(net,nIters=train["n_iters"])
        predProb = net.predict(self.xTest)
        yHat = np.where(predProb < 0.5, 0, 1)
        acc = net.acc(yHat, net.label_encoding(self.yTest))
        print("Test Accuracy %.2f" % acc)
        pass