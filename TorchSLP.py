# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------
import matplotlib
matplotlib.use("TkAgg")
import gzip # pour décompresser les données
import pickle  # pour désérialiser les données
import numpy as np # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import scipy.special
import re
from mpl_toolkits.mplot3d import axes3d, Axes3D

# fonction qui va afficher l'image située à l'index index
def affichage(image,label):
    # on redimensionne l'image en 28x28
    image = image.reshape(28,28)
    # on récupère à quel chiffre cela correspond (position du 1 dans label)
    label = np.argmax(label)
    # on crée une figure
    plt.figure()
    # affichage du chiffre
    # le paramètre interpolation='nearest' force python à afficher chaque valeur de la matrice sans l'interpoler avec ses voisines
    # le paramètre cmap définit l'échelle de couleur utilisée (ici noire et blanc)
    plt.imshow(image,interpolation='nearest',cmap='binary')
    # on met un titre
    plt.title('chiffre '+str(label))
    # on affichage les figures créées
    plt.show()

############################################################################################################################################

class Graph:
    def __init__(self):
        self.xplot = []
        self.yplot = []
        fig = plt.figure()
        self.axe = fig.add_subplot(111)
        self.line, = self.axe.plot([], [])
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    def updateGraph(self):
        self.line.set_data(self.xplot, self.yplot)
        self.axe.relim()
        self.axe.autoscale_view()
        plt.draw()
        plt.pause(0.1)

class SLP:
    def __init__(self):
        self.weights = weightReducFactor * torch.randn(input,output,device=device).type(dtype)
        self.biais = torch.ones((1,output))

    def setImage(self,image,label):
        self.x = image
        self.t = label

    def guess_real_number(self,y_pred):
        global correct
        if torch.argmax(self.t) == torch.argmax(y_pred):
            correct = correct + 1

    def train(self):
        y_pred = self.x.mm(self.weights).add(self.biais)
        # Only for graph update
        self.guess_real_number(y_pred)

        self.weights += lr * self.x.t()*(self.t - y_pred)
        self.biais += lr * (self.t - y_pred)

    def evaluate(self):
        y_pred = self.x.mm(self.weights).add(self.biais)
        self.guess_real_number(y_pred)

def test_with_multiple_parameters():
    global reducW,weightReducFactor,lr,reducLr,lock
    for i in range(19):
        if i == 10:
            reducW = reducW*0.1
        weightReducFactor = weightReducFactor - reducW
        for j in range(40):
            if j== 10 or j == 19 or j == 28 or j==37:
                reducLr = reducLr * 0.1
            lr = lr - reducLr

            # Training part
            print("Training...")
            Slp = SLP()
            for e in range(epoch):
                info = "weightReducFactor:  {}, lr: {}, epoch: {}".format(weightReducFactor, lr, e+1)
                k = 0
                correct = 0
                print(info)
                for image,label in train_loader:
                    Slp.setImage(image,label)
                    Slp.train()
                    if k==5000 and e == 0:
                        print("% : ", (correct / k) * 100)
                    if k == 5000 and (correct/k)*100 < 15 and e == 0:
                        lock = 1
                        break

                    k = k +1
                if lock == 1:
                    break
                if lock == 0:
                    # Testing part
                    if isGraphActive:
                        GraphPrecision = Graph()
                    k = 0
                    correct = 0
                    print("Test...")
                    for image in test_loader:
                        label = test_data_label[k]
                        Slp.setImage(image,label)
                        Slp.evaluate()
                        k = k + 1
                    res = "Test : weightReducFactor:  {}, lr: {}, epoch: {}, Resultat: {}%".format(round(weightReducFactor,4),round(lr,6), e+1, round((correct / len(test_data)) * 100,6))
                    #print(res)
                    outRes = open("testResultSLP.txt","a")
                    outRes.write(res)
                    outRes.write("\n")
                    outRes.close()
                    print("Nombre de prono correct : ", correct)
                    print("Pourcentage de réussite : ", (correct / len(test_data)) * 100)
            lock = 0
        lr = 0.11
        reducLr=0.01
    weightReducFactor=1.1
    reducW=0.1

def parse_res():

    with open('testResultSLP.txt') as f:
        data = []
        for line in f:
            transac = re.findall(r"[-+]?\d*\.\d+|\d+",line)
            data.append(transac)

    for j in range(1,4):
        weightRed = []
        learningRate = []
        res = []
        for i in data:
            #print("i",i[2])
            #print("j",str(j))
            if str(j) == i[2]:
                weightRed.append(float(i[0]))
                learningRate.append(float(i[1]))
                res.append(float(i[3]))

        bestParams = res.index(max(res))
        print("Meilleur paramètres pour une epoch de ", j, " sont :", "weiRed= ",weightRed[bestParams], " lr=", learningRate[bestParams], ". Résultat obtenu:", res[bestParams], "%!")
        #plt.xlabel("learningRate/WeighRed")
        #plt.ylabel("Resultats")
        #plt.scatter(learningRate,res)
        #plt.savefig("lrResScatEpoch_{}".format(j))
        #plt.show()
        #plt.xlabel("weightRed")
        #plt.ylabel("Resultats")
        #plt.scatter(weightRed,res)
        #plt.savefig("weightResScatEpoch_{}".format(j))
        #plt.show()

        #res = np.array([res,res])
        #xy = [weightRed,learningRate]
        #yx = [learningRate,weightRed]
        #fig = plt.figure()
        #ax = Axes3D(fig)
        #surf = ax.plot_surface(xy,yx,res,cmap=matplotlib.cm.coolwarm,linewidth=0,antialiased=False)
        #fig.colorbar(surf,shrink=0.5,aspect=5)
        #plt.savefig("LrWeightResEpoch_{}".format(j))
        #plt.show(block=False)
        #plt.pause(0.1)

############################################################################################################################################

# c'est ce qui sera lancé lors que l'on fait python tuto_python.py
if __name__ == '__main__':
    # nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
    TRAIN_BATCH_SIZE = 1
    # on charge les données de la base MNIST
    data = pickle.load(gzip.open('mnist.pkl.gz'), encoding='latin1')
    # images de la base d'apprentissage
    train_data = torch.Tensor(data[0][0])
    # labels de la base d'apprentissage
    train_data_label = torch.Tensor(data[0][1])
    # images de la base de test
    test_data = torch.Tensor(data[1][0])
    # labels de la base de test
    test_data_label = torch.Tensor(data[1][1])
    # on crée la base de données d'apprentissage (pour torch)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
    # on crée la base de données de test (pour torch)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
    # on crée le lecteur de la base de données d'apprentissage (pour torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # on crée le lecteur de la base de données de test (pour torch)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    # 10 fois
    #for i in range(0,10):
    # on demande les prochaines données de la base
    #   (_,(image,label)) = enumerate(train_loader).__next__()
    # on les affiche
    #  affichage(image.numpy(),label.numpy())
    # NB pour lire (plus proprement) toute la base (ce que vous devrez faire dans le TP) plutôt utiliser la formulation suivante


    #for image,label in train_loader:
    #    affichage(image.numpy(),label.numpy())
    #for image,label in test_loader:
    #    affichage(image.numpy(),label.numpy())

    dtype = torch.FloatTensor
    device = torch.device("cpu")

    isGraphActive = False
    input = 784
    output = 10

    lr = 0.0003
    epoch = 2
    weightReducFactor = 0.02

    lock = 0
    reducW = 0.1
    reducLr = 0.01

    # Function for making test and graphs
    #parse_res()
    #test_with_multiple_parameters()

    # Training part
    print("Training...")
    Slp = SLP()
    for e in range(epoch):
        info = "weightReducFactor:  {}, lr: {}, epoch: {}".format(weightReducFactor, lr, e+1)
        if isGraphActive:
            GraphPrecision = Graph()
        k = 0
        correct = 0
        print(info)
        for image,label in train_loader:
            Slp.setImage(image,label)
            Slp.train()
            if isGraphActive:
                if k % 100 == 0 and k > 0:
                    GraphPrecision.xplot.append(k)
                    GraphPrecision.yplot.append((correct / k)*100)
                    GraphPrecision.updateGraph()
            k = k +1

    # Testing part
    if isGraphActive:
        GraphPrecision = Graph()
    k = 0
    correct = 0
    print("Test...")
    for image in test_loader:
        label = test_data_label[k]
        Slp.setImage(image,label)
        Slp.evaluate()

        if isGraphActive:
            if k % 100 == 0 and k > 0:
                GraphPrecision.xplot.append(k)
                GraphPrecision.yplot.append((correct / k) * 100)
                GraphPrecision.updateGraph()
        k = k + 1
    print("Nombre de prono correct : ", correct)
    print("Pourcentage de réussite : ", (correct / len(test_data)) * 100)

