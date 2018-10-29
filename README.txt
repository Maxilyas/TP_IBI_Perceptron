GOMES Antoine
TOUATI Gaïs


################
######INFO######
################

Notre projet tourne sous python 3.X
Si vous voulez faire tourner notre projet en version 2.X il va falloir changer le chargement des données dans la base "data"
puisque le package Pickle est géré différemment entre python2 et python3
####
/!\ Nous avons utilisé la version COMPLETE de mnist (mnist.pkl.gz)
####
Vous pouvez modifier plusieurs paramètres:

	- lr : learning rate du réseau de neurone
	- epoch : Nombre de fois que tourne le réseau de neurone sur la base mnist
	- weightReducFactor : Variable permettant de diminuer ou augmenter les poids initiaux
	- hidden_layer : Nombre de neurones dans notre couche cachée (MLP.py et DMLP.py seulement)
	- nbLayers : Nombre de couche cachée (uniquement dans DMLP.py)
	- activationFunction : Choix entre sigmoid et ReLU fonction d'activité. (uniquement dans DMLP.py)
	- isGraphActive : Vous permet de voir ou pas a progression de l'apprentissage lors de l'entrainement
	- copy : Vous permet de sauvegarder dans un fichier vos résultats

################
#####LANCER#####
################

Si vous voulez lancer un des trois fichiers il vous suffit d'avoir la base de données mnist.pkl.gz ainsi que le fichier en question (TorchSLP.py,TorchMLP.py,DMLP.py)
Les paramètres ont été réglé sur un des meilleurs résultats obtenu lors des tests.



