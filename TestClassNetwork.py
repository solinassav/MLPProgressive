import numpy as np
from sklearn.datasets import load_iris
from Network import Network
import os
from Trainer import Trainer
## prendo un dataset di test dalla repository di sklearn
np.random.seed(0)
X, Y = load_iris(10000)
## Ricordare di cambiare i path
ROOT_DIR = os.path.abspath(os.curdir)
jsonStructureDir = ROOT_DIR + "\structure.txt"
jsonTrainDir = ROOT_DIR + "\\train.txt"
jsonProgressiveTrainDir = ROOT_DIR + "\\train.txt"

#TODO: leggere note per il professore
#Tutte le modalità di allenamento vengono interrotte se la rete converge
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


# esempio di test: classe Network stand-alone effettua solo un training di tutta la reta secondo le informazioni contenute nel json "structure.txt"
netTest_1 = Network(jsonStructureDir, XStandAlone= X, YStandAlone = Y)
Network.train(netTest_1)
predProb = netTest_1.predict()
yHat = np.where(predProb < 0.5, 0, 1)
acc = netTest_1.acc(yHat)
print("Test Accuracy %.2f" % acc)


# esempio di test classe Trainer + Network (simple train) (file "structure.txt" +train.txt)
trainer = Trainer(jsonTrainDir)
xTrain, xTest, yTrain, yTest = trainer.split(X = X, Y = Y)
netTest_2 = Network(jsonStructureDir, x=xTrain, y=yTrain)
trainer.simpleTrain(netTest_2)
netTest_2.saveNetwork(ROOT_DIR + "\\netTest_2")

