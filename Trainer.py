import json
import os
import random
import shutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from MLP import Network
import numpy as np


class Trainer(object):

    def __init__(self, jsonTrainDir='', jsonTrainProgDir=''):
        self.getJsonData(jsonTrainDir, jsonTrainProgDir)

    def getJsonData(self, jsonTrainDir, jsonTrainProgDir=''):
        self.isProgressive = False
        self.isSimple = False
        self.trainPlane = []
        if (jsonTrainProgDir != ''):
            self.isProgressive = True
            with open(jsonTrainProgDir, "r") as file:
                netData = file.read().replace("\n", "")
            netDataDict = json.loads(netData)
            self.trainPlane = netDataDict["train_plane"]
            pass
        else:
            self.isSimple = True
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
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(X, Y, test_size=self.testSize,
                                                                  random_state=self.randomState)
        return self.xTrain, self.xTest, self.yTrain, self.yTest

    def createNet(self, X, Y,jsonStructureDir):
        self.split(X,Y)
        self.net = Network(jsonStructureDir, x=self.xTrain, y=self.yTrain)
        return self.net

    def simpleTrain(self,  net = None, nConvergence=0):
        if (net == None):
            net = self.net
        if (nConvergence == 0):
            nConvergence = self.nIters
        Network.train(net, nIters=self.nIters, nConvergence=nConvergence)
        return self.summary(net)

    def progressiveTrain(self, net = None):
        if(net == None):
            net = self.net
        for train in self.trainPlane:
            net.unfreezAllTrainableVar()
            for layer in train["freez_layers"]:
                net.freezTrainableVar(layer)
            Network.train(net, nIters=train["n_iters"])

        return self.summary(net)

    def statistic(self,pred, net, yHat, xTest, positive = []):
        if yHat == []:
            yHat = self.yHat
        self.phiDeltaM = []
        self.acc, self.trueVector, self.oneHotEncodedYTest = net.acc(yHat, self.yTest, positive)
        self.phiDeltaM.append(self.computePhiDelta(self.trueVector, yHat, xTest, self.oneHotEncodedYTest))
        for layerOutput in pred:
            self.phiDeltaM.append(self.computePhiDelta(self.trueVector, yHat, layerOutput, self.oneHotEncodedYTest))
        print("Test Accuracy %.2f" % self.acc)
        return self.acc, self.phiDeltaM

    def summary(self, net, yHat=[], xTest=[]):
        self.pred = net.getAll(self.xTest)
        self.pred_array = self.pred[self.pred.__len__()-1]
        self.yHat = np.where(self.pred.pop(self.pred.__len__()-1) < 0.5, 0, 1)
        if yHat == []:
            yHat = self.yHat
        if xTest == []:
            xTest = self.xTest
        acc, phiDeltaM = self.statistic(self.pred, net, yHat, self.xTest.transpose())
        self.summaryToPDf(net, acc, phiDeltaM)
        return self.acc, phiDeltaM

    def computePhiDelta(self, trueVector, yHat, xTest, yTest):
        dim = len(trueVector)
        ratio = []
        phiM = []
        deltaM = []
        for xT in xTest:
            xT -= xT.min()
            if(xT.max()!=0):
                xT /= xT.max()
            isOneLabel = (len(yHat.transpose()) == 2)
            phi = []
            delta = []
            j=0
            for yH in yHat.transpose():
                if not isOneLabel:
                    TN = 0
                    P = 0
                    N = 0
                    TP = 0
                    FN = 0
                    FP = 0
                    for i in range(0, dim - 1):
                        if(len(yTest.shape)>1):
                            yT = yTest.transpose().__getitem__(j)
                        else:
                            yT = yTest
                        if(yH.__getitem__(i)):
                            P+=1
                            if(yT.__getitem__(i)):
                                TP += xT.__getitem__(i)
                        else:
                            N+=1
                            if (not yT.__getitem__(i)):
                                TN += 1- xT.__getitem__(i)
                    if (P == 0):
                        TP = 0
                    else:
                        TP = TP / P
                    if (N == 0):
                        TN = 0
                    else:
                        TN = TN / N
                    delta.append(TP + TN - 1)
                    phi.append(TP - TN)
                    j += 1
                else:
                    j+=1
                    isOneLabel = False

            if (len(phi) > 0):
                phiM.append(phi)
                deltaM.append(delta)
        return self.phiDelta(phiM, deltaM)

    def phiDelta(self, phiM, deltaM):
        phiDeltaM = []
        for i in range(0, len(phiM.__getitem__(0))):
            phiDelta = []
            for j in range(0, len(phiM)):
                phiDelta.append([phiM.__getitem__(j).__getitem__(i), deltaM.__getitem__(j).__getitem__(i)])
            phiDeltaM.append(phiDelta)
        return phiDeltaM

    def summaryToPDf(self, model, acc, phiDeltaM):
        nInput = model.nInputDim
        nOutput = model.nOutputDim
        nHiddenLayer = model.nHidden
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        nLayer = nHiddenLayer.__len__()
        layerSpace = np.where(nLayer > 3, (nLayer) * 3 / nLayer, 3)
        scala = max(max(nHiddenLayer), nInput)
        directory = "Networks/" + str(self.networkName) + "/Summary"
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        path = directory + "/" + str(self.networkName) + "Summary.tex"
        file = open(path, "w")
        file.write("\documentclass[a4paper]{report}\n" +
                   "\\usepackage{tikz}\n" +
                   "\\usepackage{pgfplots}\n")
        file.write("\\begin{document}\n" +
                   "\\author{MLP Solinas}\n" +
                   "\\title{" + model.networkName +
                   "\\\\ structure, training, performance}\n" +
                   "\maketitle\n" +
                   "\pagestyle{plain}\n" +
                   "\def\layersep{" + str(layerSpace) + "}\n")
        ## struttura
        file.write("\\chapter{Network structure}\n")
        file.write("\\begin{tikzpicture}[shorten >=1pt,->,draw=black!50, node distance=\layersep]\n" +
                   "\t\\tikzstyle{every pin edge}=[<-,shorten <=1pt]\n" +
                   "\t\\tikzstyle{neuron}=[circle,fill=black!25,minimum size=" + str(
            np.where(scala > 15, 17 * 15 / scala, 17)) + "pt,inner sep=0pt]\n" +
                   "\t\\tikzstyle{input neuron}=[neuron, fill=green!50];\n" +
                   "\t\\tikzstyle{output neuron}=[neuron, fill=red!50];\n" +
                   "\t\\tikzstyle{hidden neuron}=[neuron, fill=blue!50];\n" +
                   "\t\\tikzstyle{annot} = [text width=4em, text centered]\n")
        file.write("\t\\foreach \\name / \y in {1,...," + str(nInput) + "}\n" +
                   "\t\t\\node[input neuron,pin = {[pin edge={<-}]left:}] (I-\\name) at (0,-\y*" + str(
            np.where(scala > 15, 15 / scala, 1)) + ") {};\n")
        for i in range(0, nLayer):
            if (i == 0):
                diff = np.where(scala > 15, (nHiddenLayer.__getitem__(i) - nInput) * 15 / scala,
                                (nHiddenLayer.__getitem__(i) - nInput))
            else:
                diff += np.where(scala > 15,
                                 (nHiddenLayer.__getitem__(i) - nHiddenLayer.__getitem__(i - 1)) * 15 / scala,
                                 (nHiddenLayer.__getitem__(i) - nHiddenLayer.__getitem__(i - 1)))
            file.write("\t\\foreach \\name / \y in {1,...," +
                       str(nHiddenLayer.__getitem__(i)) + "}\n" +
                       "\t\t \path[yshift=" + str(diff / 2) +
                       "cm]\n" +
                       "\t\t\tnode[hidden neuron] (" + alphabet.__getitem__(i) +
                       "-\\name) at (" + str(i + 1) +
                       "*" + "\layersep  ,-\y *" + str(np.where(scala > 15, 15 / scala, 1)) + "cm) {};\n")
        diff += np.where(scala > 15, (nOutput - nHiddenLayer.__getitem__(nLayer - 1)) * 15 / scala,
                         (nOutput - nHiddenLayer.__getitem__(nLayer - 1)))
        file.write("\t\\foreach \\name / \y in {1,...," + str(nOutput) + "}\n" +
                   "\t\t \path[yshift=" + str(diff / 2) + "cm]\n" +
                   "\t\t\tnode[output neuron] (O-\\name) at (" + str(
            nLayer + 1) + "*" + "\layersep  ,-\y*" + str(
            np.where(scala > 15, 15 / scala, 1)) + " cm - \layersep) {};\n")
        file.write("\t\\foreach \source in {1,...," +
                   str(nInput) + "}\n" +
                   "\t\t\\foreach \dest in {1,...," +
                   str(nHiddenLayer.__getitem__(0)) + "}\n" +
                   "\t\t\t\path (I-\source) edge (" +
                   str(alphabet.__getitem__(0)) + "-\dest );\n")
        for i in range(1, nLayer):
            file.write("\t\\foreach \source in {1,...," +
                       str(nHiddenLayer.__getitem__(i - 1)) + "}\n" +
                       "\t\t\\foreach \dest in {1,...," +
                       str(nHiddenLayer.__getitem__(i)) + "}\n" +
                       "\t\t\t\path (" + str(alphabet.__getitem__(i - 1)) +
                       "-\source) edge (" +
                       alphabet.__getitem__(i) +
                       "-\dest );\n")
        file.write("\\foreach \source in {1,...," +
                   str(nHiddenLayer.__getitem__(nLayer - 1)) + "}\n" +
                   "\t\\foreach \dest in {1,...," + str(nOutput) + "}\n" +
                   "\t\t\path (" + str(alphabet.__getitem__(nLayer - 1)) +
                   "-\source) edge (O-\dest );\n")
        file.write("\end{tikzpicture}\\\\\n")
        file.write("The network is realized with " + str(nLayer) + " hidden layers respectively with ")
        for n in nHiddenLayer:
            file.write(str(n) + ", ")
        file.write("neurons.\\\\ \n")
        file.write("Activation function for hidden layers: " + model.hiddenActivation + "\\\\ \n")
        file.write("Activation function for output layer: " + model.outputActivation + "\\\\ \n")

        file.write("\section{Use model}\n")
        file.write("For see weights and biases in csv format go to MLPProgressive/" + model.getPathForNetwork() + "\\\\ \n")
        file.write(
            "For used the model you must go in the path of the project where you want to use the model and use command:\\\\"
            + " \\\\1) mkdir " + model.getPathForNetwork() + "\\\\"
            + " \\\\2) mkdir Config\\\\"
            + "\\\\3) Now copy the csv files in the WB folder and structure.txt in the Config folder (of the project where you want to use the model)\\\\"
            + "\\\\4) Import Network class in the project\\\\"
            + "\\\\5) Create the network: \\\\ net = Network(jsonStructureDir,rebuild=True)\\\\"
            + "\\\\6) Use the network: \\\\ predProb = net.predict(input)\\\\ (input must be a list) \\\\")
        file.write("\\\\net.predict(input) return, for each input, an array containing the probability for each label provided during training."
                   + " The order of the probabilities is maintained with respect to the order in which the labels were supplied during the training phase\\\\")
        ##Training
        file.write("\chapter{Training}\n")
        if self.isSimple:
            file.write("You've trained the network in one epoch obtainig an accuracy of " +
                        str(self.acc) +" in " + str(self.nIters) + " iters " + "using the " + model.nameOptimizer +"\\\\\n")
            file.write("Learning rate: " + str(self.learningRate) + "\\\\\n")
            file.write("Random state: " + str(self.randomState) + "\\\\\n")
            file.write("Test size: " + str(
                self.testSize) + " (The test size is the percentual of the dataset used for testing network)\\\\\n")
        elif self.isProgressive:
            file.write("You've trained the network in " + str(len(self.trainPlane)) +
                       " epoch obtainig an accuracy of " + str(self.acc) + " using the " + model.nameOptimizer +"\\\\\n")
            file.write("Learning rate: " + str(self.learningRate) + "\\\\\n")
            file.write("Random state: " + str(self.randomState) + "\\\\\n")
            file.write("Test size: " + str(
                self.testSize) + " (The test size is the percentual of the dataset used for testing network)\\\\\n")
            i=0
            for epoch in self.trainPlane:
                i+=1
                file.write("Epoch" + str(i)+ ":\\\\\n")
                file.write("Freezed Layer:\t")
                for fl in epoch.get("freez_layers"):
                    file.write(str(fl) +", \t")
                file.write("for " + str(epoch.get("n_iters")) + "iters \\\\")
        ##PhiDelta
        file.write("\\chapter{Phi Delta}")
        file.write("\\begin{tikzpicture}" +
                   "\\begin{axis}[" +
                   "\taxis lines = left,\n" +
                   "\txlabel = $\phi$,\n" +
                   "\tylabel = {$\delta$},\n" +
                   "\tlegend pos = outer north east\n" +
                   "]\n" +
                   "\\addplot [\n" +
                   "\tdomain=0:1,\n" +
                   "\tsamples=100,\n" +
                   "\tcolor=black,\n" +
                   "\tforget plot,\n" +
                   "]\n" +
                   "{-x +1 };\n" +
                   "\\addplot [\n" +
                   "\tdomain=-1:0,\n" +
                   "\tsamples=100,\n"
                   "\tcolor=black,\n" +
                   "\tforget plot,\n" +
                   "]\n" +
                   "{-x - 1 };\n" +
                   "\\addplot [\n" +
                   "\tsamples=100,\n" +
                   "\tdomain=-1:0,\n" +
                   "\tcolor=black,\n" +
                   "\tforget plot,\n" +
                   "]\n" +
                   "{ x + 1 };\n" +
                   "\\addplot [\n" +
                   "\tdomain=0:1,\n" +
                   "\tsamples=100,\n" +
                   "\tcolor=black,\n" +
                   "\tforget plot,\n" +
                   "]\n" +
                   "{x - 1 };\n" +
                   "\\addplot[only marks, mark  options={purple}] table\n" +
                   "{\n" +
                   "-1 0\n" +
                   "};\n" +
                   "\\addlegendentry\n" +
                   "{$Dummy\tNo$}\n" +
                   "\\addplot[only marks, mark  options={blue}] table\n" +
                   "{\n" +
                   "1 0\n" +
                   "};\n" +
                   "\\addlegendentry\n" +
                   "{$Dummy\tYes$}\n" +
                   "\\addplot[only marks, mark  options={red}] table\n" +
                   "{\n" +
                   "0 -1\n" +
                   "};\n" +
                   "\\addlegendentry\n" +
                   "{$Always\tWrong$}\n" +
                   "\\addplot[only marks, mark  options={green}] table\n" +
                   "{\n" +
                   "0 1\n" +
                   "};\n" +
                   "\\addlegendentry\n" +
                   "{$Oracle$}\n" +
                   ")\n" +
                   "\end{axis}\n" +
                   "\end{tikzpicture}\\\\\n")
        file.write(
            "The data are enclosed in a \"diamond\" defined by the equation $\mid$$\phi$$\mid$ + $\mid$$\delta$$\mid$ = 1 which represents the measurement domain." +
            "For every output a diagram containing a set of points, each referring to a different feature, is obtained." +
            "If the position of a point tends towards the point (1, 0) then the represented feature tends to shift the " +
            "clasfer's response towards the positive class (dummy yes), in the same way if the point tends towards (-1, 0)" +
            "the represented feature tends to shift the outcome to the negative class (dummy no). " +
            "In this way, we can say that we have a better classifier when the value of $\phi$ is closer to 0." +
            "If the position of a point tends towards the point (0, 1) then the represented feature tends to shift the " +
            "outcome towards the correct class (Oracle), in the same way if the point tends towards (0, -1)" +
            " the feature represented tends to shift the outcome to the wrong class (always wrong)." +
            "In this way, we can say that we have a better classifier when the value of $\delta$ is closer to the unit.\\\\")

        n_l = 0
        for layer in phiDeltaM:
            i = 1
            for graph in layer:
                if(n_l == 0):
                    file.write("\section{Label " + str(i) + ", Input}")
                else:
                    file.write("\section{Label " + str(i) + ", Hidden Layer " + str(n_l) + "}")
                i += 1
                file.write("\\begin{tikzpicture}" +
                           "\\begin{axis}[" +
                           "\taxis lines = left,\n" +
                           "\txlabel = $\phi$,\n" +
                           "\tylabel = {$\delta$},\n" +
                           "\tlegend pos = outer north east\n" +
                           "]\n" +
                           "\\addplot [\n" +
                           "\tdomain=0:1,\n" +
                           "\tsamples=100,\n" +
                           "\tcolor=black,\n" +
                           "\tforget plot,\n" +
                           "]\n" +
                           "{-x +1 };\n" +
                           "\\addplot [\n" +
                           "\tdomain=-1:0,\n" +
                           "\tsamples=100,\n"
                           "\tcolor=black,\n" +
                           "\tforget plot,\n" +
                           "]\n" +
                           "{-x - 1 };\n" +
                           "\\addplot [\n" +
                           "\tsamples=100,\n" +
                           "\tdomain=-1:0,\n" +
                           "\tcolor=black,\n" +
                           "\tforget plot,\n" +
                           "]\n" +
                           "{ x + 1 };\n" +
                           "\\addplot [\n" +
                           "\tdomain=0:1,\n" +
                           "\tsamples=100,\n" +
                           "\tcolor=black,\n" +
                           "\tforget plot,\n" +
                           "]\n" +
                           "{x - 1 };\n")
                j = 1
                for feature in graph:
                    r = random.random()
                    g = random.random()
                    b = random.random()
                    file.write("\\addplot[only marks, color ={rgb:red," + str(r) + ";green," + str(g) + ";blue," + str(
                        b) + "}] table\n" +
                               "{\n" + str(feature.__getitem__(0)) + " " + str(feature.__getitem__(1)) +
                               "\n" +
                               "};\n" +
                               "\\addlegendentry\n" +
                               "{$Feature\t" + str(j) + "$}\n")
                    j += 1
                file.write("\end{axis}\n" +
                           "\end{tikzpicture}\\\\\n")
            n_l += 1
        file.write("\end{document}\n")
        file.close()
        os.chdir(directory)
        os.system("pdflatex " + str(self.networkName) + "Summary.tex")
        os.chdir(model.rootDir)

