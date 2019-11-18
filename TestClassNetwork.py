import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from Network import Network
import os
from Trainer import Trainer

#TODO: leggere note per il professore
#As-Is:
    #Ogni uscita corrisponde a un label, le uscite sono poste a 1 se la probabilità per quel label è maggiore di 0.5, a 0 diversamente.
    #In questo caso l'input può essere attribuito a più classi, a una classe o a nessuna classe
    #Se la rete ha un output (tipo vero/falso), ne vedrete due a causa della codifica dei label (esempio, vero:(1,0) falso:(0,1)).
#To-Be:
    #Come vorrebbe che si utilizzi la rete nel caso in cui un input non possa appartenere a più classi contemporaneamente?
        #(propongo di aggiungere il campo "mutually_exclusive" : true/false nel json structure.txt)
        #l'input verrà considerato appartenente solo alla classe con la probabilità più alta
    #Come vorrebbe che si utilizzi la rete nel caso in cui un input non possa non appartenere a nessuna classe?
        #(propongo di aggiungere il campo "must_to_be classificated")
    #Entrambe le soluzioni sono già disponibili nel mia copia locale del codice, deve solo decidere come vuole usarle
    #Propongo un metodo predictReal che renda le uscite della rete (come numeri reali) e non una classificazione, in modo da dare
    # all'utente la possibilità di gestire la classificazione ignorando questi ponti

## prendo un dataset di test dalla repository di sklearn
np.random.seed(0)
data, labels = load_breast_cancer(50000)

## Ricordare di cambiare i path dentro la radice che sceglierete dovrete creare la cartella config con dentro i json, consiglio di lasciare tutto come lo fornisco io
ROOT_DIR = os.path.abspath(os.curdir)
os.chdir(ROOT_DIR)
jsonStructureDir = ROOT_DIR + "\config\structure.txt"
jsonTrainDir = ROOT_DIR + "\config\\train.txt"
jsonProgressiveTrainDir = ROOT_DIR + "\config\\progressiveTrain.txt"

#1 esempio di test: classe Network stand-alone effettua solo un training di tutta la rete secondo le informazioni contenute nel json "structure.txt"
netTest_1 = Network(jsonStructureDir, XStandAlone= data, YStandAlone = labels)
Network.train(netTest_1)
predProb = netTest_1.predict()
yHat = np.where(predProb < 0.5, 0, 1)
acc, trueVector, oneHotYTest = netTest_1.acc(yHat)
print("Test Accuracy %.2f" % acc)

#2 esempio di test classe Trainer + Network (simple train) (file "structure.txt" +train.txt)
trainer = Trainer(jsonTrainDir)
xTrain, xTest, yTrain, yTest = trainer.split(X = data, Y = labels)
netTest_2 = Network(jsonStructureDir, x=xTrain, y=yTrain)
trainer.simpleTrain(netTest_2)

#3 esempio di test classe Trainer + Network con allenamento progressivo
trainer_2 = Trainer(jsonTrainDir, jsonProgressiveTrainDir)
xTrain, xTest, yTrain, yTest = trainer_2.split(X = data, Y = labels)
netTest_3 = Network(jsonStructureDir, x=xTrain, y=yTrain)
trainer_2.progressiveTrain(netTest_3)

#4 esempio di test classe Trainer stand-alone con allenamento semplice
trainer_3 = Trainer(jsonTrainDir)
trainer_3.createNet(data, labels, jsonStructureDir)
trainer_3.simpleTrain()
trainer_3.net.saveNetwork()

#5 esempio di test classe Trainer stand-alone con allenamento progressivo
trainer_3 = Trainer(jsonTrainDir, jsonProgressiveTrainDir)
trainer_3.createNet(data, labels, jsonStructureDir)
trainer_3.progressiveTrain()
trainer_3.net.saveNetwork()

#6 esempio di test classe Trainer stand-alone con allenamento semplice e riutilizzo rete
trainer_4 = Trainer(jsonTrainDir)
net_test_4 = trainer_4.createNet(data, labels, jsonStructureDir)
trainer_4.simpleTrain()
net_test_4.saveNetwork()

#7 esempio di test classe Trainer stand-alone con allenamento progressivo e riutilizzo rete
trainer_5 = Trainer(jsonTrainDir, jsonProgressiveTrainDir)
net_test_5 = trainer_5.createNet(data, labels, jsonStructureDir)
trainer_5.progressiveTrain()
net_test_5.saveNetwork()

#8 Uso della rete salvata, per usare una rete salvata basta mettere il suo nome nel file jsonStructureDir e istanzianziarla con rebuild
netTest_6=Network(jsonStructureDir,rebuild=True)
predProb = netTest_6.predict(xTest)
yHat = np.where(predProb < 0.5, 0, 1)
acc, trueVector, oneHotYTest = netTest_6.acc(yHat,yTest)
print("Test Accuracy %.2f" % acc)
