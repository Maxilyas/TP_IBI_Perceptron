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
        self.weights_h1 = torch.randn(input, hidden_input)
        self.weights_h2 = torch.randn(hidden_input, output)
        self.biais1 = torch.ones((1, hidden_input))
        self.biais2 = torch.ones((1, output))

    def setImage(self):
        self.x = image
        self.t = label

    def compareGuessRealNumber(self,outY2):
        global correct
        if torch.argmax(outY2) == torch.argmax(self.t):
            correct = correct + 1

    def reformat_label_outY(self,outY2):
        for i in range(output):
            if i != torch.argmax(outY2):
                outY2[0][i] = 0
            else:
                outY2[0][i] = 1

    def train(self):

        # Activity hidden layer
        sumWX = self.x.mm(self.weights_h1).add(self.biais1)
        outY1 = torch.sigmoid(sumWX)
        # Activity out layer
        outY2 = outY1.mm(self.weights_h2).add(self.biais2)

        # Gradient descent out layer
        self.reformat_label_outY(outY2)
        error2 = (self.t.sub(outY2))

        # Compare for GRAPH UPDATE
        Mlp.compareGuessRealNumber(outY2)

        # Gradient descent hidden layer
        error1 = outY1 * ( 1 - outY1)*error2.mm(self.weights_h2.t())
        # Weights correction for each images
        self.weights_h1 += lr * self.x.t().mm(error1)
        self.biais1 += lr * error1
        self.weights_h2 += lr * outY1.t().mm(error2)
        self.biais2 += lr * error2

    def evaluate(self):
        # Activity hidden layer
        sumWX = self.x.mm(self.weights_h1).add(self.biais1)
        outY1 = torch.sigmoid(sumWX)
        # Activity out layer
        outY2 = outY1.mm(self.weights_h2).add(self.biais2)

        # Compare to get accuracy and graph update
        Mlp.compareGuessRealNumber(outY2)




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

    len_train_data = len(train_data)
    len_test_data = len(test_data)
###########################################################
    # Params
    #  95.557 with lr = 0.059 epoch = 4 and hidden_input = 256
    #  96,457 with lr = 0.059 epoch = 4 and hidden_input = 1024
    input = 784
    hidden_input = 256
    output = 10
    lr = 0.059
    epoch = 4

###########################################################

    # Training part
    Mlp = MLP()
    for e in range(epoch):
        correct = 0
        GraphPrecision = Graph()
        k = 0
        print("Epoch number : ", e)
        for image,label in train_loader:
            Mlp.setImage()
            Mlp.train()
            # Update graph when k=100 images
            if k % 100 == 0 and k > 0:
                print("Training Image : ", k)
                GraphPrecision.dynGraph()
            k = k + 1

    # Testing part

    GraphPrecision = Graph()
    correct = 0
    k = 0
    for image in test_loader:
        label = test_data_label[k]
        Mlp.setImage()
        Mlp.evaluate()

        # Update graph each k=100 images
        if k % 100 == 0 and k > 0:
            print("Processing Image : ", k)
            GraphPrecision.dynGraph()

        k = k + 1

    print("Nombre de prono correct : ", correct)
    print("Pourcentage de réussite : ", (correct / len_test_data) * 100)


