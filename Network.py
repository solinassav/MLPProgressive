import os
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import json
class network(object):
    def __init__(self, X, Y,json_structure_dir):
        with open(json_structure_dir, 'r') as file:
            net_data = file.read().replace('\n', '')
        net_data_dict = json.loads(net_data)
        self.hidden_activation = net_data_dict["hidden_activation"]
        self.output_activation = net_data_dict["output_activation"]
        self.name_optimizer = net_data_dict["optimizer"]
        self.networkName= net_data_dict["network_name"]
        self.sess = tf.Session()
        self.X = X
        self.Y = Y
        self.n_output_dim = self.Y.shape[1]
        self.n_inputs = X.shape[0]
        self.n_input_dim = X.shape[1]
        self.n_output = self.Y.shape[0]
        self.learning_rate = net_data_dict["learning_rate"]
        self.n_hidden = net_data_dict["n_hidden_list"]
        self.n_layer = len(self.n_hidden)
        self.trainable_var = []
        self.trainable_var.append(tf.trainable_variables())
        self.X_input, self.y, self.logits, self.cost = self.build_layer(self.n_output_dim, self.n_layer, self.hidden_activation, self.output_activation)
        self.trainable_var = tf.trainable_variables()
        if self.name_optimizer == "AdamOptimizer" or self.name_optimizer == "":
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        if self.name_optimizer == "AdagradOptimizer":
            self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.learning_rate)
        if self.name_optimizer == "AdadeltaOptimizer":
            self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.learning_rate)
        if self.name_optimizer == "GradientDescentOptimizer":
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.cost,var_list= self.trainable_var)

    def build_layer(self,n_output,n_layer = 1, hidden_activation = "relu", output_activation="sigmoid"):

        print('Now i''m building layers')
        initializer = tf.contrib.layers.xavier_initializer()
        X_input = tf.compat.v1.placeholder(tf.float32,
                                           [None, self.n_input_dim],
                                           name='input')
        y = tf.compat.v1.placeholder(tf.float32,
                                     [None, n_output], name='y')
        layer = []
        if hidden_activation=="relu" or hidden_activation == "":
            activation_h = tf.nn.elu
        if hidden_activation == "sigmoid":
            activation_h = tf.nn.sigmoid
        if hidden_activation == "softmax":
            activation_h =tf.nn.softmax
        hidden1 = fully_connected(X_input, self.n_hidden.__getitem__(0),
                                  activation_fn=activation_h,
                                  weights_initializer=initializer)
        layer.insert(0, hidden1)
        print('layer: 1, numero_nodi: '+ str(self.n_hidden.__getitem__(0)))
        for i in range(1,n_layer):
            print('layer: '+ str(i+1) + ', numero_nodi: ' + str(self.n_hidden.__getitem__(i)))
            layer.insert(i,fully_connected(layer.__getitem__(i-1), self.n_hidden.__getitem__(i),
                                           activation_fn=activation_h,
                                           weights_initializer=initializer))

        print(tf.trainable_variables())
        if output_activation=="relu":
            activation_o = tf.nn.elu
        if output_activation == "softmax":
            activation_o = tf.nn.softmax
        if output_activation == "sigmoid" or output_activation == "":
            activation_o = tf.nn.sigmoid
        logits = fully_connected(layer.__getitem__(n_layer-1), self.n_output_dim,
                                 activation_fn=activation_o,
                                 weights_initializer=initializer)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                       logits=logits)
        cost = tf.reduce_mean(loss)
        return X_input, y, logits, cost

    def train_one_iter(self):
        _, cost = self.sess.run([self.train_op, self.cost],
                                feed_dict={
                                    self.X_input: self.X,
                                    self.y: self.Y})
        return cost

    def predict(self, X_test):
        pred = self.sess.run([self.logits],
                             feed_dict={
                                 self.X_input: X_test})[0]
        return pred

    def train(model, n_iters=1000):
        print('I''m learning, wait for '+ str(n_iters) + ' iterates')
        model.sess.run(tf.global_variables_initializer())
        cost = []
        for i in range(n_iters):
            print('iterate'+ str(i))
            actual_cost = model.train_one_iter()
            print('cost: ' + str(actual_cost))
            cost.append(actual_cost)
        return cost

    def summary(model, acc, cost):
        n_input=model.n_input_dim
        n_output=model.n_output_dim
        n_hidden_layer=model.n_hidden
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        layerspace = max(n_hidden_layer) /3
        n_layer = n_hidden_layer.__len__()
        file = open("copy.tex", "w")
        file.write("\documentclass[article]{report}\n" +
                   "\\usepackage{tikz}\n" +
                   "\\begin{document}\n" +
                    "\\author{MLP Solinas}\n"+
	                "\\title{"+ model.networkName +"}\n"+
	                "\maketitle\n"+
                   "\pagestyle{empty}\n" +
                   "\def\layersep{" + str(layerspace) + "}\n" +
                    "\\newpage\n"+
                    "\section{Structure and data}\n" +
                   "\\begin{tikzpicture}[shorten >=1pt,->,draw=black!50, node distance=\layersep]\n" +
                    "\centering\n"+
                   "\t\\tikzstyle{every pin edge}=[<-,shorten <=1pt]\n" +
                   "\t\\tikzstyle{neuron}=[circle,fill=black!25,minimum size=17pt,inner sep=0pt]\n" +
                   "\t\\tikzstyle{input neuron}=[neuron, fill=green!50];\n" +
                   "\t\\tikzstyle{output neuron}=[neuron, fill=red!50];\n" +
                   "\t\\tikzstyle{hidden neuron}=[neuron, fill=blue!50];\n" +
                   "\t\\tikzstyle{annot} = [text width=4em, text centered]\n")
        file.write("\t\\foreach \\name / \y in {1,...," + str(n_input) + "}\n" +
                   "\t\t\\node[input neuron,pin = {[pin edge={<-}]left:}] (I-\\name) at (0,-\y) {};\n")
        for i in range(0, n_layer):
            if (i == 0):
                diff = n_hidden_layer.__getitem__(i) - n_input
            else:
                diff += n_hidden_layer.__getitem__(i) - n_hidden_layer.__getitem__(i - 1)
            alphabet_ = alphabet.__getitem__(i)
            file.write("\t\\foreach \\name / \y in {1,...," + str(n_hidden_layer.__getitem__(i)) + "}\n" +
                       "\t\t \path[yshift=" + str(diff / 2) + "cm]\n" +
                       "\t\t\tnode[hidden neuron] (" + alphabet_ + "-\\name) at (" + str(
                i + 1) + "*" + "\layersep  ,-\y cm) {};\n")
        diff += n_output-n_hidden_layer.__getitem__(n_layer - 1)
        file.write("\t\\foreach \\name / \y in {1,...," + str(n_output) + "}\n" +
                   "\t\t \path[yshift=" + str(diff / 2) + "cm]\n" +
                   "\t\t\tnode[output neuron] (O-\\name) at (" + str(
            n_layer + 1) + "*" + "\layersep  ,-\y cm - \layersep) {};\n")
        file.write("\t\\foreach \source in {1,...," + str(n_input) + "}\n" +
                   "\t\t\\foreach \dest in {1,...," + str(n_hidden_layer.__getitem__(0)) + "}\n" +
                   "\t\t\t\path (I-\source) edge (" + str(alphabet.__getitem__(0)) + "-\dest );\n")
        for i in range(1, n_layer):
            file.write("\t\\foreach \source in {1,...," + str(n_hidden_layer.__getitem__(i - 1)) + "}\n" +
                       "\t\t\\foreach \dest in {1,...," + str(n_hidden_layer.__getitem__(i)) + "}\n" +
                       "\t\t\t\path (" + str(alphabet.__getitem__(i - 1)) + "-\source) edge (" + alphabet.__getitem__(
                i) + "-\dest );\n")
        file.write("\\foreach \source in {1,...," + str(n_hidden_layer.__getitem__(n_layer - 1)) + "}\n" +
                   "\t\\foreach \dest in {1,...," + str(n_output) + "}\n" +
                   "\t\t\path (" + str(alphabet.__getitem__(n_layer - 1)) + "-\source) edge (O-\dest );\n")
        file.write("\section{Train}\n")

        file.write("\end{tikzpicture}\n" +
                   "\end{document}\n")
        file.close()
        os.system("pdflatex copy.tex")
        return 0




