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

class SLP:
    def __init__(self,image,label):
        self.x = image.numpy()
        self.t = label.numpy()

        self.outY_2 = np.zeros((1, output))


    def activity_function_out_layer(self,weights_h2,biais_w2):
        self.outY_2 = np.dot(self.x.reshape(1,784), weights_h2) + biais_w2
        return self.outY_2

    def guess_real_number(self):
        global correct
        if np.argmax(self.outY_2) == np.argmax(self.t):
            correct = correct + 1

    def correction_weights(self,weights_h2,biais_w2):

        weights_h2 += lr * np.dot(self.x.reshape(784,1),np.subtract(self.t , self.outY_2))
        biais_w2 += lr * np.subtract(self.t, self.outY_2)

        return weights_h2,biais_w2



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

    input = 784
    output = 10

    biais_h2 = np.ones((1,output))
    biais_w2 = np.random.rand(1, output)
    weights_h2 = np.random.rand(input, output)

    lr = 0.01
    epoch = 1

    # Training part

    for e in range(epoch):
        correct = 0
        GraphPrecision = Graph()
        k = 0
        print("Epoch number : ", e)
        for image,label in train_loader:
            Slp = SLP(image,label)
            print("Training Image : ",k)

            Slp.outY_2 = Slp.activity_function_out_layer(weights_h2,biais_w2)

            Slp.guess_real_number()
            if k % 100 == 0 and k > 0:
                GraphPrecision.xplot.append(k)
                GraphPrecision.yplot.append((correct / k)*100)
                GraphPrecision.updateGraph()

            weights_h2, biais_w2 = Slp.correction_weights(weights_h2,biais_w2)
            k = k +1

    # Testing part

    GraphPrecision = Graph()
    correct = 0
    k = 0
    for k in range(len_test_data):
        image = test_data[k]
        label = test_data_label[k]
        Slp = SLP(image, label)
        print("Processing Image : ", k)

        Slp.outY_2 = Slp.activity_function_out_layer(weights_h2, biais_w2)

        Slp.guess_real_number()
        if k % 100 == 0 and k > 0:
            GraphPrecision.xplot.append(k)
            GraphPrecision.yplot.append((correct / k) * 100)
            GraphPrecision.updateGraph()
        k = k + 1

    print("Nombre de prono correct : ", correct)
    print("Pourcentage de réussite : ", (correct / len_test_data) * 100)


