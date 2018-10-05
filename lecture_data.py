# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip # pour décompresser les données
import pickle  # pour désérialiser les données
import numpy as np # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data


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


    len_train_data = len(train_data)
    len_test_data = len(test_data)

    # SLP !
    biais = 1
    weights = np.random.rand(784, 10)


    # MLP
    biais_h1 = 1
    weights_h1 = np.random.rand(784, 256)
    biais_h2 = 1
    weights_h2 = np.random.rand(256, 10)

    lr = 0.1
    epoch = 1
    correct = 0

    # Training part Simple Layer Perceptron
    #for e in range(epoch):
    #    print("Epoch number : ",e)
    #    for k in range(len_train_data):
    # x est l'entrée de la kieme image
    #        x = train_data[k]
    # t est le label de la kieme image
    #        t = train_data_label[k]
    # reset des sorties de la kieme image
    #        outY = np.zeros(10)
    # Pronostique de la sortie
    #        for i in range(10):
    #            for j in range(784):
    #                outY[i] = outY[i] + weights[j][i]*float(x[j])
    #            outY[i] = outY[i] + biais
    #        prono = np.argmax(outY)
    #        values,indices =torch.max(t,0)


    #print("Prono : ", prono)
    #print("Vrai Chiffre : ",indices)

    # Si sortie incorrecte, correction des poids
    #        if prono != indices:
    #            for i in range(10):
    #                if i != prono:
    #                    outY[i] = 0
    #                else:
    #                    outY[i] = 1
    #                lab = int(t[i])
    #                for j in range(784):
    #                    weights[j][i] += lr * float(x[j]) * (lab - outY[i])
    #                biais += lr * (lab - outY[i])
    #print("Prono : ", outY[i])
    #print("Real One ; ",t[i])

    #    print("Weights after training : ", weights)



    #Training Part : Multiple Layer Perceptron
    for e in range(epoch):
        print("Epoch number : ", e)
        for k in range(len_train_data):
            # x est l'entrée de la kieme image
            x = train_data[k]
            # t est le label de la kieme image
            t = train_data_label[k]
            # reset des sorties de la kieme image

            outY_1 = np.zeros(256)
            outY_2 = np.zeros(10)

            error_2 = np.zeros(10)
            error_1 = np.zeros(256)
            # Activité couche caché
            for i in range(256):
                for j in range(784):
                    try:
                        outY_1[i] = outY_1[i] + 1 / (1 + np.exp(-(weights_h1[j][i] * float(x[j]))))
                    except:
                        print("OUTY_1 : ",outY_1[i][j] )
                #outY_1[i] = outY_1[i] + biais_h1

            #Activité couche sortie
            for i in range(10):
                for j in range(256):
                    outY_2[i] = outY_2[i] + weights_h2[j][i] * outY_1[j]
                #outY_2[i] = outY_2[i] + biais_h2

            prono_Y2 = np.argmax(outY_2)
            values, indices = torch.max(t, 0)
            print("Prono :", prono_Y2)
            print("REAL NUMBER : ",indices )
            if prono_Y2 != indices:
                for i in range(10):
                    if i != prono_Y2:
                        outY_2[i] = 0
                    else:
                        outY_2[i] = 1
                #Retro propag du gradient
                for i in range(10):
                    error_2[i] = int(t[i]) - outY_2[i]
                    for j in range(256):
                        error_1[j] = error_1[j] + outY_1[j] * (1 - outY_1[j]) * error_2[i] * weights_h2[j][i]

                for i in range(256):
                    for j in range(784):
                        weights_h1[j][i] = weights_h1[j][i] + lr * error_1[i]*float(x[j])
                    #biais_h1 = biais_h1 + lr*error_1[i]
                for i in range(10):
                    for j in range(256):
                        weights_h2[j][i] = weights_h2[j][i] + lr*error_2[i]*outY_1[j]
                    #biais_h2 = biais_h2 + lr * error_2[i]
                    # print("Prono : ", outY[i])
                    # print("Real One ; ",t[i])

        print("Weights after training : ", weights)





    # Testing part
    #for image,label in test_loader:
    #    affichage(image.numpy(),label.numpy())

    for k in range(len_test_data):
        print("Image processing : ",k)
        x = test_data[k]
        t = test_data_label[k]
        outY = np.zeros(10)
        for i in range(10):
            for j in range(784):
                outY[i] = outY[i] + weights[j][i]*float(x[j])
            outY[i] = outY[i] + biais
        prono = np.argmax(outY)
        values,indices = torch.max(t,0)

        #Tensor_outY = torch.from_numpy(outY)
        if prono == indices:
            correct = correct + 1

    print("Nombre de prono correct : ", correct)
    print("Pourcentage de réussite : ", (correct/len_test_data)*100)


