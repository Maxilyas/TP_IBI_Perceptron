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

class MLP:
    def __init__(self,image,label):
        self.x = image.numpy()
        self.t = label.numpy()

        self.outY_1 = np.zeros((1, hidden_input))
        self.outY_2 = np.zeros((1, output))

        self.error_2 = np.zeros((1, output))
        self.error_1 = np.zeros((1, hidden_input))

    def activity_function_hidden_layer(self, weights_h1, biais_w1):
        #print("Weights_h1 : ",weights_h1)
        sumWX = np.dot(self.x.reshape(1,784), weights_h1) + biais_w1
        #print("SUMWX : ",sumWX)
        for i in range(hidden_input):
            self.outY_1[0][i] = scipy.special.expit(sumWX[0][i])
            #self.outY_1[0][i] = 1 / (1 + np.exp(-sumWX[0][i]))
        #print ("OUTY1 ; ",self.outY_1)

        return self.outY_1

    def activity_function_out_layer(self,weights_h2,biais_w2):
        self.outY_2 = np.dot(self.outY_1, weights_h2) + biais_w2
        return self.outY_2

    def guess(self):
        return np.argmax(self.outY_2)

    def real_number(self):
        return np.argmax(self.t)
        #return torch.argmax(t)

    def reformat_label_outY(self,prono_Y2):
        for i in range(output):
            if i != prono_Y2:
                self.outY_2[0][i] = 0
            else:
                self.outY_2[0][i] = 1
        return self.outY_2

    def gradient_descent(self, weights_h2):

        self.error_2 = np.subtract(self.t, self.outY_2)
        sumEW = (np.dot(self.error_2,weights_h2.T))
        #self.error_1 = np.dot((np.dot(self.outY_1,(np.subtract(1, self.outY_1.T)))), sumEW.T)
        self.error_1 = np.multiply(np.multiply(self.outY_1, (np.subtract(1, self.outY_1))), sumEW)
        #self.error_1 = sumEW.T

        #print("ERROR 1 : ",self.error_1)
        #print("GRADIENT")
        #print("SHAPE SUMEW :",sumEW.shape)
        #print("SHAPE OUTY1",self.outY_1.shape)
        #print("SHAPE T OUTY1",(self.outY_1.T).shape)
        #print("SHAPE MULTI",np.multiply(self.outY_1 , (np.subtract(1, self.outY_1))).shape)
        #print("SHAPE ERROR 1",self.error_1.shape)
        #print("SHAPE ERROR 2",self.error_2.shape)

        return self.error_2,self.error_1

    def correction_weights(self,weights_h1,biais_w1,weights_h2,biais_w2):

        #print("Shape X : ",self.x.reshape)
        #print("SHAPE CORR ERROR 1",self.error_1.shape)
        #print("SHAPE WEIGHT H1 : ",weights_h1.shape)
        #print("Biais h1",biais_h1.shape)
        #print("WEIGHTS SHAPE H2 ",weights_h2.shape)

        weights_h1 += lr * np.dot(self.x.reshape(784, 1), self.error_1)
        biais_w1 = lr * self.error_1
        weights_h2 += lr * np.dot(self.outY_1.T, self.error_2)
        biais_w2 = lr * self.error_2

        return weights_h1, biais_w1, weights_h2, biais_w2



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
    hidden_input = 128
    output = 10

    biais_h1 = np.ones((1,hidden_input))
    biais_w1 = np.random.rand(1,hidden_input)
    weights_h1 = np.random.rand(input, hidden_input)
    biais_h2 = np.ones((1,output))
    biais_w2 = np.random.rand(1, output)
    weights_h2 = np.random.rand(hidden_input, output)

    lr = 0.01
    epoch = 2


    # Training part

    for e in range(epoch):
        correct = 0
        GraphPrecision = Graph()
        k = 0
        print("Epoch number : ", e)
        for image,label in train_loader:
            Mlp = MLP(image,label)
            print("Training Image : ",k)
            Mlp.outY_1 = Mlp.activity_function_hidden_layer(weights_h1,biais_w1)
            Mlp.outY_2 = Mlp.activity_function_out_layer(weights_h2,biais_w2)

            prono_Y2 = Mlp.guess()
            indices = Mlp.real_number()
            if prono_Y2 == indices:
                correct = correct + 1
            if k % 100 == 0 and k > 0:
                GraphPrecision.xplot.append(k)
                GraphPrecision.yplot.append((correct / k)*100)
                GraphPrecision.updateGraph()

            #print("Prono :", prono_Y2)
            #print("REAL NUMBER : ",indices)

            if prono_Y2 != indices:
                Mlp.outY_2 = Mlp.reformat_label_outY(prono_Y2)
                Mlp.error_2,Mlp.error_1 = Mlp.gradient_descent(weights_h2)
                weights_h1, biais_w1, weights_h2, biais_w2 = Mlp.correction_weights(weights_h1,biais_w1,weights_h2,biais_w2)
            k = k + 1

    # Testing part

    GraphPrecision = Graph()
    correct = 0
    k = 0
    for k in range(len_test_data):
        image = test_data[k]
        label = test_data_label[k]
        Mlp = MLP(image, label)
        print("Processing Image : ", k)

        Mlp.outY_1 = Mlp.activity_function_hidden_layer(weights_h1, biais_w1)
        Mlp.outY_2 = Mlp.activity_function_out_layer(weights_h2, biais_w2)

        prono_Y2 = Mlp.guess()
        indices = Mlp.real_number()

        print("Prono :", prono_Y2)
        print("REAL NUMBER : ", indices)

        if prono_Y2 == indices:
            correct = correct + 1

        if k % 100 == 0 and k > 0:
            GraphPrecision.xplot.append(k)
            GraphPrecision.yplot.append((correct / k) * 100)
            GraphPrecision.updateGraph()
        k = k + 1

    print("Nombre de prono correct : ", correct)
    print("Pourcentage de réussite : ", (correct / len_test_data) * 100)


