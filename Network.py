import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import fully_connected
import json
import numpy as np

class Network(object):

    def __init__(self, jsonStructureDir, x = [], y = [], XStandAlone = [],YStandAlone = []):
        self.getJsonData(jsonStructureDir)
        self.processDimensions(x,y,XStandAlone,YStandAlone)
        self.buildNetwork()

    def processDimensions(self,x,y,X,Y):
        self.xInput = x
        self.yInput = y
        if x == []:
            self.xInput, self.xTest, self.yInput, self.yTest = train_test_split(X, Y, test_size=self.testSize,
                                                                      random_state=self.randomState)
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
        self.trainableVar = []
        # TODO self.trainableVar deve diventare un dizionario in cui la chiave è il numero del layer
        self.trainableVar.append(tf.trainable_variables())
        self.xPlaceholder, self.yPlaceholder, self.logits, self.cost = self.buildLayer(self.nOutputDim, self.nLayer,
                                                                                       self.hiddenActivation,
                                                                                       self.outputActivation)
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
        self.layer.insert(0, fully_connected(xPlaceholder, self.nHidden.__getitem__(0), activation_fn=hiddenActivationFunction,
                                  weights_initializer=initializer))
        print("layer: 1, numero_nodi: " + str(self.nHidden.__getitem__(0)))
        for i in range(1, nLayer):
            print("layer: " + str(i + 1) + ", numero_nodi: " + str(self.nHidden.__getitem__(i)))
            self.layer.insert(i, fully_connected(self.layer.__getitem__(i - 1), self.nHidden.__getitem__(i),
                                            activation_fn=hiddenActivationFunction, weights_initializer=initializer))
        print(tf.trainable_variables())
        if outputActivation == "relu":
            outputActivationFunction = tf.nn.elu
        if outputActivation == "softmax":
            outputActivationFunction = tf.nn.softmax
        if outputActivation == "sigmoid" or outputActivation == "":
            outputActivationFunction = tf.nn.sigmoid
        logits = fully_connected(self.layer.__getitem__(nLayer - 1),
                                 self.nOutputDim,
                                 activation_fn=outputActivationFunction,
                                 weights_initializer=initializer)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=yPlaceholder,
                                                       logits=logits)
        cost = tf.reduce_mean(loss)
        return (xPlaceholder, yPlaceholder, logits, cost)

    def trainForOneIter(self):
        (_, cost) = self.sess.run([self.trainOp, self.cost],
                                  feed_dict={self.xPlaceholder: self.xInput,
                                             self.yPlaceholder: self.yInput})
        return cost

    def predict(self, xTest = []):
        if xTest == [] :
            xTest = self.xTest
        pred = self.sess.run([self.logits],
                             feed_dict={self.xPlaceholder: xTest})[0]
        return pred

    def train(model, nIters = 0):
        if nIters == 0 :
            nIters = model.nIters
        print("Im learning, wait for " + str(nIters) + " iterates")
        model.sess.run(tf.global_variables_initializer())
        cost = []
        for i in range(nIters):
            actualCost = model.trainForOneIter()
            print("Iterate: " + str(i) +" Cost: " + str(actualCost))
            cost.append(actualCost)
        return cost

    def acc(self, yHat, yTest = []):
        if yTest == []:
            yTest = self.yTest
        acc = 0
        if (len(yTest.shape) > 1):
            for i in range(0, yHat.shape[0]):
                match = 1
                match = (yHat[i][0] == yTest[i][0])
                for j in range(1, yHat.shape[1]):
                    match = (yHat[i][j] == yTest[i][j]) & match
                if (match):
                    acc += 1
            acc = acc / yHat.shape[0]
        else:
            acc = np.sum(yTest.reshape(-1, 1) == yHat) / len(yTest)
        return acc

    def getTrainableVar(self):
        #Restituisce le variabili allenabili
        return self.trainableVar

    def saveNetwork(self,folder = "\\netTest_2"):
        print(tf.trainable_variables())
        for trainableVar in tf.trainable_variables() :
            print(self.sess.run( trainableVar))

    def freezTrainableVar(self, layer):
        # TODO Questo metodo freeza un layer, rimuovendo pesi e bias dalle variabiali allenabili
        # ci si riferirà alle variabili tramite il numero del layer (crescente dall'input all'output)
        pass
    def unfreezTrainableVar(self):
        # TODO Riaggiunge a self.trainableVar i pesi e i bias
        return 0



