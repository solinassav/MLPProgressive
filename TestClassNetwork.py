import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
from Network import Network
import os
## get Dataset from repository
np.random.seed(0)
X, Y = make_moons(500, noise=0.2)

ROOT_DIR = os.path.abspath(os.curdir)
json_structure_dir = ROOT_DIR+"\structure.txt"
with open(json_structure_dir, 'r') as file:
    net_data = file.read().replace('\n', '')
net_data_dict = json.loads(net_data)
test_size = net_data_dict["test_size"]
random_state = net_data_dict["random_state"]
n_iters = net_data_dict["n_iters"]
Y = np.row_stack((Y, Y))
Y= np.transpose(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                    test_size = test_size,
                                                    random_state = random_state)
tf.reset_default_graph()
net = Network( X_train, Y_train,json_structure_dir)
cost = Network.train(net, n_iters)
pred_prob = net.predict(X_test)
y_hat = np.where(pred_prob < 0.5, 0, 1)
acc = 0
for i in range(0,y_hat.shape[0]):
    match = 1
    match = (y_hat[i][0] == Y_test[i][0])
    for j in range(1,y_hat.shape[1]):
        match = (y_hat[i][j] == Y_test[i][j]) & match
    if(match):
        acc+=1
acc=acc/y_hat.shape[0]
print("Test Accuracy %.2f" % acc)