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

    def dynGraph(self):
        self.xplot.append(k)
        self.yplot.append((correct / k) * 100)
        self.updateGraph()


class MLP:
    def __init__(self):
        self.weights_h1 = weightReducFactor * torch.randn(input, hidden_input,device=device).type(dtype)
        self.weights_h2 = torch.randn(hidden_input, output,device=device).type(dtype)
        self.biais1 = torch.ones((1, hidden_input))
        self.biais2 = torch.ones((1, output))

    def setImage(self,image,label):
        self.x = image
        self.t = label

    def compareGuessRealNumber(self,y_pred):
        global correct
        if torch.argmax(y_pred) == torch.argmax(self.t):
            correct = correct + 1

    def train(self):

        # Activity hidden layer
        sumWX = self.x.mm(self.weights_h1).add(self.biais1)
        y = torch.sigmoid(sumWX)
        # Activity out layer
        y_pred = y.mm(self.weights_h2).add(self.biais2)

        # Gradient descent out layer
        error2 = (self.t - y_pred)

        # Compare for GRAPH UPDATE
        Mlp.compareGuessRealNumber(y_pred)

        # Gradient descent hidden layer
        error1 = y * (1 - y)*error2.mm(self.weights_h2.t())
        # Weights correction for each images
        self.weights_h1 += lr * self.x.t().mm(error1)
        self.biais1 += lr * error1
        self.weights_h2 += lr * y.t().mm(error2)
        self.biais2 += lr * error2

    def evaluate(self):
        # Activity hidden layer
        sumWX = self.x.mm(self.weights_h1).add(self.biais1)
        y = torch.sigmoid(sumWX)
        # Activity out layer
        y_pred = y.mm(self.weights_h2).add(self.biais2)

        # Compare to get accuracy and graph update
        Mlp.compareGuessRealNumber(y_pred)







############################################################################################################################################

def test_with_multiple_parameters():
    global reducW,weightReducFactor,lr,reducLr,lock,hidden_input
    for i in range(19):
        if i == 10:
            reducW = reducW * 0.1
        weightReducFactor = weightReducFactor - reducW
        for j in range(40):
            if j == 10 or j == 19 or j == 28 or j == 37:
                reducLr = reducLr * 0.1
            lr = lr - reducLr
            for z in range(4):
                if z != 0:
                    hidden_input = hidden_input * 2
                # Training part
                Mlp = MLP()
                for e in range(epoch):
                    info = "weightReducFactor:  {}, lr: {}, hidden_input: {}, epoch: {}".format(weightReducFactor, lr, hidden_input, e + 1)
                    print(info)
                    correct = 0
                    k = 0
                    for image, label in train_loader:
                        Mlp.setImage(image,label)
                        Mlp.train()
                        if k == 5000 and e == 0:
                            print("% : ", (correct / k) * 100)
                        if k == 5000 and (correct / k) * 100 < 15 and e == 0:
                            lock = 1
                            break
                        k = k + 1
                    if lock == 1:
                        break

                    # Testing part
                    correct = 0
                    k = 0
                    for image in test_loader:
                        label = test_data_label[k]
                        Mlp.setImage(image,label)
                        Mlp.evaluate()

                        k = k + 1
                    res = "Test : weightReducFactor:  {}, lr: {}, hidden_input: {}, epoch: {}, Resultat: {}%".format(
                        round(weightReducFactor, 4), round(lr, 6), hidden_input, e + 1,
                        round((correct / len(test_data)) * 100, 6))
                    # print(res)
                    outRes = open("testResultMLP.txt", "a")
                    outRes.write(res)
                    outRes.write("\n")
                    outRes.close()
                    print("Nombre de prono correct : ", correct)
                    print("Pourcentage de réussite : ", (correct / len(test_data)) * 100)
                lock = 0
            hidden_input = 16
        lr = 0.11
        reducLr = 0.01
    weightReducFactor = 1.1
    reducW = 0.1

def parse_res():

    with open('testResultMLP.txt') as f:
        data = []
        for line in f:
            transac = re.findall(r"[-+]?\d*\.\d+|\d+",line)
            data.append(transac)
    hidden = 16
    for z in range(4):
        for j in range(1,5):

            weightRed = []
            learningRate = []
            res = []
            for i in data:
                #print("i",i[2])
                #print("j",str(j))
                if str(j) == i[3] and str(hidden) == i[2]:
                    weightRed.append(float(i[0]))
                    learningRate.append(float(i[1]))
                    res.append(float(i[4]))
            bestParams = res.index(max(res))
            #print("Meilleur paramètres pour une couche cachée de ",hidden," et epoch de ", j," sont :", "weiRed= ",weightRed[bestParams], " lr=", learningRate[bestParams],". Résultat obtenu:", res[bestParams],"%!")

            plt.xlabel("learningRate/WeighRed")
            plt.ylabel("Resultats")
            plt.scatter(learningRate,res)
            #plt.savefig("lrResScatEpoch_{}".format(j))
            #plt.show()
            #plt.xlabel("weightRed")
            #plt.ylabel("Resultats")
            plt.scatter(weightRed,res)
            plt.savefig("weightResScatEpoch_{}_hidden{}".format(j,hidden))
            plt.show()
        hidden = hidden*2


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
###########################################################
    # Params

    input = 784
    hidden_input = 128
    output = 10
    lr = 0.008
    epoch = 4
    weightReducFactor = 0.01


    #######################
    lock = 0
    reducW = 0.1
    reducLr = 0.01
    #######################

###########################################################
    # Function for making test and graphs
    parse_res()
    #test_with_multiple_parameters(reducW,weightReducFactor,lr,reducLr,lock, hidden_input)

    # Training part
    Mlp = MLP()
    for e in range(epoch):
        info = "weightReducFactor:  {}, lr: {}, hidden_input: {}, epoch: {}".format(weightReducFactor, lr,hidden_input, e + 1)
        print(info)
        correct = 0
        if isGraphActive:
            GraphPrecision = Graph()
        k = 0
        for image,label in train_loader:
            Mlp.setImage(image,label)
            Mlp.train()
            # Update graph when k=100 images
            if isGraphActive:
                if k % 1000 == 0 and k > 0:
                    print("Training Image : ", k)
                    GraphPrecision.dynGraph()
            k = k + 1

    # Testing part

    GraphPrecision = Graph()
    correct = 0
    k = 0
    for image in test_loader:
        label = test_data_label[k]
        Mlp.setImage(image,label)
        Mlp.evaluate()

        # Update graph each k=100 images
        if k % 1000 == 0 and k > 0:
            print("Processing Image : ", k)
            GraphPrecision.dynGraph()

        k = k + 1

    print("Nombre de prono correct : ", correct)
    print("Pourcentage de réussite : ", (correct / len(test_data)) * 100)
