import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from Network import Network
import numpy as np



class Trainer(object):

    def __init__(self, jsonTrainDir= '', jsonTrainProgDir = ''):
        self.getJsonData(jsonTrainDir, jsonTrainProgDir)


    def getJsonData(self,jsonTrainDir, jsonTrainProgDir = ''):
        self.trainPlane = []
        if ( jsonTrainProgDir !='' ):
            with open(jsonTrainProgDir, "r") as file:
                netData = file.read().replace("\n", "")
            netDataDict = json.loads(netData)
            self.trainPlane =  netDataDict["train_plane"]
            pass
        else:
            with open(jsonTrainDir, "r") as file:
                netData = file.read().replace("\n", "")
            netDataDict = json.loads(netData)
            self.nIters = netDataDict["n_iters"]
        self.path = netDataDict["path"]
        self.datasetName = netDataDict["d_name"]
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
        self.predProb = net.predict(self.xTest)
        self.yHat = np.where(self.predProb < 0.5, 0, 1)
        return self.summary(net)


    def progressiveTrain(self,net):
        for train in self.trainPlane:
            net.unfreezAllTrainableVar()
            for layer in train["freez_layers"]:
                net.freezTrainableVar(layer)
            Network.train(net,nIters=train["n_iters"])
        self.predProb = net.predict(self.xTest)
        self.yHat = np.where(self.predProb < 0.5, 0, 1)
        return self.summary(net)



    def statistic(self, net, yHat):
        if yHat == []:
            yHat = self.yHat
        acc, trueVector = net.acc(yHat, self.yTest)
        phi, delta, ratio = self.phiDeltaRatio(trueVector,yHat)
        print("Test Accuracy %.2f" % acc)
        return acc, phi, delta, ratio

    def summary(self, net, yHat = []):
        if yHat == []:
            yHat = self.yHat
        acc, phi, delta, ratio = self.statistic(net, yHat)
        return acc, phi, delta, ratio


    def phiDeltaRatio(self, trueVector, yHat):
        dim = len(trueVector)
        ratio = []
        phi = []
        delta = []
        for yH in yHat.transpose():
            w00 = 0
            w01 = 0
            w10 = 0
            w11 = 0
            for i in range(0,dim-1):
                if (trueVector.__getitem__(i)):
                    if(yH.__getitem__(i)==1):
                        w11+=1
                    else:
                        w00+=1
                else:
                    if(yH.__getitem__(i)==1):
                        w01+=1
                    else:
                        w10+=1
            tp = w11/(w11+w10) if (w11+w10) > 0. else 0
            fp = w10/(w11+w10) if (w11+w10) > 0. else 0
            if (tp != 0) & (fp != 0):
                np = w11 + w10
                nf = w01 + w00
                ratio.append(nf/np)
                phi.append(np + fp -1)
                delta.append( 2*tp - (np + fp -1) - 1)
            else :
                ratio.append(0)
                phi.append(0)
                delta.append(0)
        return phi, delta, ratio


