import os
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import json


class Network(object):
    def __init__(self, x, y, jsonStructureDir):

        with open(jsonStructureDir, "r") as file:
            netData = file.read().replace("\n", "")
        netDataDict = json.loads(netData)
        self.hiddenActivation = netDataDict["hidden_activation"]
        self.outputActivation = netDataDict["output_activation"]
        self.nameOptimizer = netDataDict["optimizer"]
        self.networkName = netDataDict["network_name"]
        self.sess = tf.Session()
        self.xInput = x
        self.yInput = y
        self.nOutputDim = self.yInput.shape[1]
        self.nInputs = x.shape[0]
        self.nInputDim = x.shape[1]
        self.nOutput = self.yInput.shape[0]
        self.learningRate = netDataDict["learning_rate"]
        self.nHidden = netDataDict["n_hidden_list"]
        self.nLayer = len(self.nHidden)
        self.trainableVar = []
        self.trainableVar.append(tf.trainable_variables())
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
        self.trainOp = self.optimizer.minimize(self.cost, var_list=self.trainableVar)

    def buildLayer(self, nOutput, nLayer=1, hiddenActivation="", outputActivation="sigmoid"):

        hiddenActivationFunction = tf.nn.elu
        print("Now im building layers")
        initializer = tf.contrib.layers.xavier_initializer()
        xPlaceholder = tf.compat.v1.placeholder(tf.float32, [None, self.nInputDim], name="input")
        yPlaceholder = tf.compat.v1.placeholder(tf.float32, [None, nOutput], name="y")
        layer = []
        if hiddenActivation == "sigmoid":
            hiddenActivationFunction = tf.nn.sigmoid
        if hiddenActivation == "softmax":
            hiddenActivationFunction = tf.nn.softmax
        layer.insert(0, fully_connected(xPlaceholder, self.nHidden.__getitem__(0), activation_fn=hiddenActivationFunction,
                                  weights_initializer=initializer))
        print("layer: 1, numero_nodi: " + str(self.nHidden.__getitem__(0)))
        for i in range(1, nLayer):
            print("layer: " + str(i + 1) + ", numero_nodi: " + str(self.nHidden.__getitem__(i)))
            layer.insert(i, fully_connected(layer.__getitem__(i - 1), self.nHidden.__getitem__(i),
                                            activation_fn=hiddenActivationFunction, weights_initializer=initializer))
        print(tf.trainable_variables())
        if outputActivation == "relu":
            outputActivationFunction = tf.nn.elu
        if outputActivation == "softmax":
            outputActivationFunction = tf.nn.softmax
        if outputActivation == "sigmoid" or outputActivation == "":
            outputActivationFunction = tf.nn.sigmoid
        logits = fully_connected(layer.__getitem__(nLayer - 1),
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

    def predict(self, xTest):
        pred = self.sess.run([self.logits],
                             feed_dict={self.xPlaceholder: xTest})[0]
        return pred

    def train(model, nIters=1000):
        print("Im learning, wait for " + str(nIters) + " iterates")
        model.sess.run(tf.global_variables_initializer())
        cost = []
        for i in range(nIters):
            print("iterate" + str(i))
            actualCost = model.trainForOneIter()
            print("cost: " + str(actualCost))
            cost.append(actualCost)
        return cost

    def summary(model, acc, cost):
        nInput = model.nInputDim
        nOutput = model.nOutputDim
        nHiddenLayer = model.nHidden
        alphabet = ["A", "B", "C", "D", "E", "F", "G", "H"]
        layerspace = max(nHiddenLayer) / 3
        nLayer = nHiddenLayer.__len__()
        file = open("copy.tex", "w")
        file.write("\documentclass[article]{report}\n"
                   + "\\usepackage{tikz}\n" + "\\begin{document}\n"
                   + "\\author{MLP Solinas}\n" + "\\title{"
                   + model.networkName + "}\n" + "\maketitle\n"
                   + "\pagestyle{empty}\n" + "\def\layersep{"
                   + str(layerspace) + "}\n" + "\\newpage\n"
                   + "\section{Structure and data}\n"
                   + "\\begin{tikzpicture}[shorten >=1pt,->,draw=black!50, node distance=\layersep]\n"
                   + "\centering\n"
                   + "\t\\tikzstyle{every pin edge}=[<-,shorten <=1pt]\n"

                   + "\t\\tikzstyle{neuron}=[circle,fill=black!25,minimum size=17pt,inner sep=0pt]\n"

                   + "\t\\tikzstyle{input neuron}=[neuron, fill=green!50];\n"

                   + "\t\\tikzstyle{output neuron}=[neuron, fill=red!50];\n"

                   + "\t\\tikzstyle{hidden neuron}=[neuron, fill=blue!50];\n"

                   + "\t\\tikzstyle{annot} = [text width=4em, text centered]\n"
                   )
        file.write("\t\\foreach \\name / \y in {1,...," + str(nInput)
                   + "}\n"
                   + "\t\t\\node[input neuron,pin = {[pin edge={<-}]left:}] (I-\\name) at (0,-\y) {};\n"
                   )
        for i in range(0, nLayer):
            if i == 0:
                diff = nHiddenLayer.__getitem__(i) - nInput
            else:
                diff += nHiddenLayer.__getitem__(i) \
                        - nHiddenLayer.__getitem__(i - 1)
            file.write(
                "\t\\foreach \\name / \\y in {1,...,{0}}\n\t\t \\path[yshift={1}cm]\n\t\t\tnode[hidden neuron] ({2}-\\name) at ({3}*\\layersep  ,-\\y cm) {};\n".format(
                    str(nHiddenLayer.__getitem__(i)), str(diff / 2), alphabet.__getitem__(i), str(i + 1)))
        diff += nOutput - nHiddenLayer.__getitem__(nLayer - 1)
        file.write(
            "\t\\foreach \\name / \\y in {1,...,{0}}\n\t\t \\path[yshift={1}cm]\n\t\t\tnode[output neuron] (O-\\name) at ({2}*\\layersep  ,-\\y cm - \\layersep) {};\n".format(
                str(nOutput), str(diff / 2), str(nLayer + 1)))
        file.write(
            "\t\\foreach \\source in {1,...,{0}}\n\t\t\\foreach \\dest in {1,...,{1}}\n\t\t\t\\path (I-\\source) edge ({2}-\\dest );\n".format(
                str(nInput), str(nHiddenLayer.__getitem__(0)), str(alphabet.__getitem__(0))))
        for i in range(1, nLayer):
            file.write(
                "\t\\foreach \\source in {1,...,{0}}\n\t\t\\foreach \\dest in {1,...,{1}}\n\t\t\t\\path ({2}-\\source) edge ({3}-\\dest );\n".format(
                    str(nHiddenLayer.__getitem__(i - 1)), str(nHiddenLayer.__getitem__(i)),
                    str(alphabet.__getitem__(i - 1)), alphabet.__getitem__(i)))
        file.write(
            "\\foreach \\source in {1,...,{0}}\n\t\\foreach \\dest in {1,...,{1}}\n\t\t\\path ({2}-\\source) edge (O-\\dest );\n".format(
                str(nHiddenLayer.__getitem__(nLayer - 1)), str(nOutput), str(alphabet.__getitem__(nLayer - 1))))
        file.write("\section{Train}\n")

        file.write("\end{tikzpicture}\n" + "\end{document}\n")
        file.close()
        os.system("pdflatex copy.tex")
        return 0
