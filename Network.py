import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import fully_connected
import json
import numpy as np
import shutil

class Network(object):


    def __init__(self, jsonStructureDir, x = [], y = [],  XStandAlone = [],YStandAlone = [], rebuild = False):
        self.rebuild = rebuild
        self.getJsonData(jsonStructureDir)
        if(self.rebuild):
            self.buildFromRepository()
        else:
            self.processDimensions(x, y, XStandAlone, YStandAlone)
            self.trainableVar = []
            self.buildNetwork()


    def processDimensions(self,x,y,X,Y):
        self.xInput = x
        self.yInput = y
        if x == []:
            self.xInput, self.xTest, self.yInput, self.yTest = train_test_split(X, Y, test_size=self.testSize, random_state=self.randomState)
            self.yTest = self.label_encoding(self.yTest)
        self.yInput = self.label_encoding(self.yInput)
        if (len(self.yInput.shape) > 1):
            self.nOutputDim = self.yInput.shape[1]
        else:
            self.nOutputDim = 1
            self.yInput = self.yInput.reshape(-1, 1)
        self.nInputs = self.xInput.shape[0]
        self.nInputDim = self.xInput.shape[1]
        self.nOutput = self.yInput.shape[0]
        self.nLayer = len(self.nHidden)


    def buildNetwork (self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.unfreezAllTrainableVar()
        self.xPlaceholder, self.yPlaceholder, self.logits, self.cost = self.buildLayer(self.nOutputDim, self.nLayer, self.hiddenActivation, self.outputActivation)
        self.trainableVar = tf.trainable_variables()
        if self.nameOptimizer == "AdamOptimizer" or self.nameOptimizer == "":
            self.optimizer = \
                tf.compat.v1.train.AdamOptimizer(self.learningRate)
        if self.nameOptimizer == "AdagradOptimizer":
            self.optimizer = \
                tf.compat.v1.train.AdagradOptimizer(self.learningRate)
        if self.nameOptimizer == "AdadeltaOptimizer":
            self.optimizer = \
                tf.compat.v1.train.AdadeltaOptimizer(self.learningRate)
        if self.nameOptimizer == "GradientDescentOptimizer":
            self.optimizer = \
                tf.compat.v1.train.GradientDescentOptimizer(self.learningRate)
        # NB basta modificare il contenuto di self.traibleVar, recuperando matrici e bias da tf.trainable_variables(),
        # per scegliere se freezare dei layer. Di default nessun layer è freezato
        self.trainOp = self.optimizer.minimize(self.cost, var_list=self.trainableVar)


    def getJsonData(self,jsonStructureDir):
        with open(jsonStructureDir, "r") as file:
            netData = file.read().replace("\n", "")
        netDataDict = json.loads(netData)
        self.hiddenActivation = netDataDict["hidden_activation"]
        self.outputActivation = netDataDict["output_activation"]
        self.nameOptimizer = netDataDict["optimizer"]
        self.networkName = netDataDict["network_name"]
        self.testSize = netDataDict["test_size"]
        self.randomState = netDataDict["random_state"]
        self.nIters = netDataDict["n_iters"]
        self.learningRate = netDataDict["learning_rate"]
        self.nHidden = netDataDict["n_hidden_list"]


    def buildLayer(self, nOutput, nLayer=1, hiddenActivation="Relu", outputActivation="sigmoid"):
        print("Now im building layers")
        initializer = tf.contrib.layers.xavier_initializer()
        xPlaceholder = tf.compat.v1.placeholder(tf.float32, [None, self.nInputDim], name="input")
        yPlaceholder = tf.compat.v1.placeholder(tf.float32, [None, nOutput], name="y")
        self.layer = []
        if hiddenActivation == "sigmoid":
            hiddenActivationFunction = tf.nn.sigmoid
        elif hiddenActivation == "softmax":
            hiddenActivationFunction = tf.nn.softmax
        else :
            hiddenActivationFunction = tf.nn.elu
        self.layer.insert(0, fully_connected(xPlaceholder, self.nHidden.__getitem__(0), activation_fn=hiddenActivationFunction, weights_initializer=initializer, biases_initializer= initializer))
        print("layer: 1, numero_nodi: " + str(self.nHidden.__getitem__(0)))
        for i in range(1, nLayer):
            print("layer: " + str(i + 1) + ", numero_nodi: " + str(self.nHidden.__getitem__(i)))
            self.layer.insert(i, fully_connected(self.layer.__getitem__(i - 1), self.nHidden.__getitem__(i), activation_fn=hiddenActivationFunction, weights_initializer=initializer, biases_initializer= initializer))
        print(tf.trainable_variables())
        if outputActivation == "relu":
            outputActivationFunction = tf.nn.elu
        if outputActivation == "softmax":
            outputActivationFunction = tf.nn.softmax
        if outputActivation == "sigmoid" or outputActivation == "":
            outputActivationFunction = tf.nn.sigmoid
        logits = fully_connected(self.layer.__getitem__(nLayer - 1), self.nOutputDim, activation_fn=outputActivationFunction, weights_initializer=initializer)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=yPlaceholder, logits=logits)
        cost = tf.reduce_mean(loss)
        return (xPlaceholder, yPlaceholder, logits, cost)


    def trainForOneIter(self):
        (_, cost) = self.sess.run([self.trainOp, self.cost], feed_dict={self.xPlaceholder: self.xInput, self.yPlaceholder: self.yInput})
        return cost


    def predict(self, xTest = []):
        pred = []
        if xTest == [] :
            xTest = self.xTest
        pred = self.sess.run([self.logits], feed_dict={self.xPlaceholder: xTest})[0]
        return pred


    def train(model, nIters = 0, nConvergence = 0):
        if nIters == 0 :
            nIters = model.nIters
        if (nConvergence == 0):
            nConvergence = model.nIters
        print("Im learning, wait for " + str(nIters) + " iterates")
        model.sess.run(tf.global_variables_initializer())
        cost = []
        for i in range(nIters):
            actualCost = model.trainForOneIter()
            print("Iterate: " + str(i) +" Cost: " + str(actualCost))
            cost.append(actualCost)
            if (model.isConverged(cost, nConvergence)):
                break
        return cost


    def isConverged(self, array, n = 100):
        if(array.__len__()<n):
            isConverged = 0
        else:
            isConverged = 1
            for i in range(2,n):
                isConverged = isConverged & (array[-i] == array[-i+1])
        return isConverged


    def acc(self, yHat, yTest = []):
        trueVector = []
        if yTest == []:
            yTest = self.yTest
        else:
            yTest = self.label_encoding(yTest)
        acc = 0
        if (len(yTest.shape) > 1):
            for i in range(0, yHat.shape[0]):
                match = 1
                match = (yHat[i][0] == yTest[i][0])
                for j in range(1, yHat.shape[1]):
                    match = (yHat[i][j] == yTest[i][j]) & match
                if (match):
                    trueVector.append(True)
                    acc += 1
                else:
                    trueVector.append(False)
            acc = acc / yHat.shape[0]
        else:
            acc = np.sum(yTest.reshape(-1, 1) == yHat) / len(yTest)
        return acc, trueVector


    def getTrainableVar(self):
        return self.trainableVar


    def label_encoding(self,array):
        oneHot = OneHotEncoder(categories='auto')
        oneHot.fit_transform(array.reshape(-1, 1))
        return oneHot.transform(array.reshape(-1, 1)).toarray()


    def weight(self,x):
        return 2*x


    def bias(self,x):
        return 2*x +1


    def freezTrainableVar(self, layer):
        self.trainableVar.remove(tf.trainable_variables().__getitem__(self.weight(layer)))
        self.trainableVar.remove(tf.trainable_variables().__getitem__(self.bias(layer)))


    def unfreezAllTrainableVar(self):
        self.trainableVar[:] = []
        for trainableVar in tf.trainable_variables():
            self.trainableVar.append(trainableVar)


    def unfreezTrainableVar(self,layer):
        self.trainableVar.append(tf.trainable_variables().__getitem__(self.weight(layer)))
        self.trainableVar.append(tf.trainable_variables().__getitem__(self.bias(layer)))


    def saveNetwork(self):
        print(tf.trainable_variables())
        i = 0
        directory = "Networks/" + str(self.networkName) + "/WB"
        shutil.rmtree(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for trainableVar in tf.trainable_variables():
            if(i%2==0):
                np.savetxt(directory + '/W'+str(i//2)+'.csv', self.sess.run( trainableVar), delimiter=",")
                #self.sess.run( trainableVar).tofile('W'+str(i//2)+'.csv',sep=',')
            else:
                np.savetxt(directory + '/B'+str(i//2)+'.csv', self.sess.run( trainableVar), delimiter=",")
            i+=1

    def getTensorFromFile(self, filePath):
        return tf.convert_to_tensor(np.loadtxt(open(filePath, "rb"), delimiter=","), dtype=tf.float32)

    def getPathForNetwork(self):
        return "Networks/{}/WB/".format(self.networkName)

    def getPathForWeight(self, fileIndex):
        return "{}W{}.csv".format(self.getPathForNetwork(), fileIndex)

    def getPathForBias(self, fileIndex):
        return "{}B{}.csv".format(self.getPathForNetwork(), fileIndex)

    def getSigmoidForIndex(self, index):
        weight = self.weights.__getitem__(index)
        bias = self.biasses.__getitem__(index)
        layer = self.layer.__getitem__(index - 1) if index != 0 else self.xPlaceholder
        sigmoid = tf.nn.sigmoid(tf.math.add(tf.matmul(layer, weight), bias))
        return sigmoid

    def getSoftmaxForIndex(self, index):
        weight = self.weights.__getitem__(index)
        bias = self.biasses.__getitem__(index)
        layer = self.layer.__getitem__(index - 1) if index != 0 else self.xPlaceholder
        softmax = tf.nn.softmax(tf.math.add(tf.matmul(layer, weight), bias))
        return softmax

    def getReluLayerForIndex(self, index):
        weight = self.weights.__getitem__(index)
        bias = self.biasses.__getitem__(index)
        layer = self.layer.__getitem__(index - 1) if index != 0 else self.xPlaceholder
        reluLayer = tf.compat.v1.nn.relu_layer(layer, weight, bias)
        return reluLayer

    def getEluLayerForIndex(self, index):
        weight = self.weights.__getitem__(index)
        bias = self.biasses.__getitem__(index)
        layer = self.layer.__getitem__(index - 1) if index != 0 else self.xPlaceholder
        eluLayer = tf.nn.elu(tf.math.add(tf.matmul(layer, weight), bias))
        return eluLayer

    def buildFromRepository(self):

        self.weights, self.biasses, self.layer = [], [], []
        self.sess = tf.Session()

        i = 0
        csvPath = self.getPathForWeight(0)
        while os.path.exists(csvPath):
            self.weights.append(self.getTensorFromFile(csvPath))
            csvPath = self.getPathForBias(i)
            self.biasses.append(self.getTensorFromFile(csvPath))
            i += 1
            csvPath = self.getPathForWeight(i)

        self.xPlaceholder = tf.compat.v1.placeholder(tf.float32, [None, self.weights.__getitem__(0).shape[0]],
                                                     name="input")

        i = 0
        weightsLength = self.weights.__len__() - 1
        while i < weightsLength:
            if self.hiddenActivation == "sigmoid":
                self.layer.append(self.getSigmoidForIndex(i))

            elif self.hiddenActivation == "softmax":
                self.layer.append(self.getSoftmaxForIndex(i))

            else:
                self.layer.append(self.getReluLayerForIndex(i))

            i += 1

        if self.outputActivation == "relu":
            self.logits = self.getEluLayerForIndex(i)

        elif self.outputActivation == "softmax":
            self.logits = self.getSoftmaxForIndex(i)

        else:
            self.logits = self.getSigmoidForIndex(i)


