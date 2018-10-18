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
        self.x = image
        self.t = label
        #self.outY_1 = np.zeros((1, hidden_input))
        #self.outY_2 = np.zeros((1, output))
        #self.error_2 = np.zeros((1, output))
        #self.error_1 = np.zeros((1, hidden_input))

    def activity_function_hidden_layer(self):
        # Activity hidden layer
        sumWX = self.x.mm(weights_h1).add(biais1)
        outY1 = torch.sigmoid(sumWX)

        return outY1
        #for i in range(hidden_input):
            #self.outY_1[0][i] = scipy.special.expit(sumWX[0][i])
        #self.outY_1 = 1 / (1 + np.exp(-sumWX))

    def activity_function_out_layer(self):
        # Activity out layer
        outY2 = outY1.mm(weights_h2).add(biais2)

        return outY2

    def compareGuessRealNumber(self):
        global correct
        if torch.argmax(outY2) == torch.argmax(self.t):
            correct = correct + 1

    def reformat_label_outY(self,outY2):
        for i in range(output):
            if i != torch.argmax(outY2):
                outY2[0][i] = 0
            else:
                outY2[0][i] = 1

    def gradient_descent(self):
        # Gradient descent out layer
        global outY2
        self.reformat_label_outY(outY2)
        error2 = (self.t.sub(outY2))


        # Gradient descent hidden layer
        #self.error_1 = self.outY_1 * (1 - self.outY_1) * sumEW

        error1 = outY1 * ( 1 - outY1)*error2.mm(weights_h2.t())
        #self.error_1 = sumEW
        return error2,error1
    def correction_weights(self):
        # Weights correction for each images
        global weights_h1, biais1, weights_h2, biais2
        weights_h1 += lr * self.x.t().mm(error1)
        biais1 += lr * error1
        weights_h2 += lr * outY1.t().mm(error2)
        biais2 += lr * error2




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

    input = 784
    hidden_input = 256
    output = 10
    lr = 0.04
    epoch = 1

    #weights_h1 = np.random.uniform(-1,1,(input,hidden_input))
    #weights_h2 = np.random.uniform(-1, 1, (hidden_input, output))
    #weights_h1 = np.random.rand(input, hidden_input)
    weights_h1 = torch.randn(input,hidden_input)
    weights_h2 = torch.randn(hidden_input,output)
    biais1 = torch.ones((1,hidden_input))
    #weights_h2 = np.random.rand(hidden_input, output)
    biais2 = torch.ones((1,output))

###########################################################
    #biais_w1 = np.random.rand(1,hidden_input)
    #biais_w2 = np.random.rand(1, output)

    # Training part

    for e in range(epoch):
        correct = 0
        GraphPrecision = Graph()
        k = 0
        print("Epoch number : ", e)
        for image,label in train_loader:
            Mlp = MLP()

            # Activity function
            outY1 = Mlp.activity_function_hidden_layer()
            outY2 = Mlp.activity_function_out_layer()

            # Gradient and weights correction
            error2,error1 = Mlp.gradient_descent()
            Mlp.correction_weights()



            Mlp.compareGuessRealNumber()
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

        Mlp = MLP()
        print("Processing Image : ", k)

        # Activation functions
        outY1 = Mlp.activity_function_hidden_layer()
        outY2 = Mlp.activity_function_out_layer()

        # Compare guess and real number
        Mlp.compareGuessRealNumber()
        # Update graph each k=100 images
        if k % 100 == 0 and k > 0:
            GraphPrecision.dynGraph()

        k = k + 1

    print("Nombre de prono correct : ", correct)
    print("Pourcentage de réussite : ", (correct / len_test_data) * 100)


