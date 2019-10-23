import numpy as np
from sklearn.datasets import load_iris
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
data, labels = load_iris(10000)

## Ricordare di cambiare i path
ROOT_DIR = os.path.abspath(os.curdir)
jsonStructureDir = ROOT_DIR + "\structure.txt"
jsonTrainDir = ROOT_DIR + "\\train.txt"
jsonProgressiveTrainDir = ROOT_DIR + "\\progressiveTrain.txt"


# esempio di test: classe Network stand-alone effettua solo un training di tutta la reta secondo le informazioni contenute nel json "structure.txt"
netTest_1 = Network(jsonStructureDir, XStandAlone= data, YStandAlone = labels)
Network.train(netTest_1)
predProb = netTest_1.predict()
yHat = np.where(predProb < 0.5, 0, 1)
acc, trueVector = netTest_1.acc(yHat)
print("Test Accuracy %.2f" % acc)


# esempio di test classe Trainer + Network (simple train) (file "structure.txt" +train.txt)
trainer = Trainer(jsonTrainDir)
xTrain, xTest, yTrain, yTest = trainer.split(X = data, Y = labels)
netTest_2 = Network(jsonStructureDir, x=xTrain, y=yTrain)
trainer.simpleTrain(netTest_2)



# esempio di test classe Trainer + Network con allenamento progressivo
trainer = Trainer(jsonTrainDir, jsonProgressiveTrainDir)
xTrain, xTest, yTrain, yTest = trainer.split(X = data, Y = labels)
netTest_3 = Network(jsonStructureDir, x=xTrain, y=yTrain)
trainer.progressiveTrain(netTest_3)
netTest_3.saveNetwork()


# Uso della rete salvata, per usare una rete salvata basta mettere il suo nome nel file jsonStructureDir e istanzianziarla con rebuild
netTest_4=Network(jsonStructureDir,rebuild=True)
predProb = netTest_4.predict(xTest)
yHat = np.where(predProb < 0.5, 0, 1)
acc, trueVector= netTest_4.acc(yHat,yTest)
print("Test Accuracy %.2f" % acc)







