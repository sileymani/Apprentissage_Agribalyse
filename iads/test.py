# Importation de librairies standards:
import numpy as np
import pandas as pd
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt  

# un nouvel import utile pour la 3D:
from matplotlib import cm


# Importation de la librairie pickle
import pickle as pkl

# lecture des donnéees en dimension 2 dans un dataframe pandas
data2D = pkl.load(open('data-projet/data-2D.pkl', 'rb')) 
X2D = np.array(data2D[['x1', 'x2']], dtype=float) # conversion de type pour une meilleure compatibilité
Y2D = np.array(data2D['label'], dtype=float)

# pour les données en dimension 5, la méthode est la même, modifier seulement les noms des colonnes

# Importation de votre librairie iads:
# La ligne suivante permet de préciser le chemin d'accès à la librairie iads
import sys
sys.path.append('../')   # iads doit être dans le répertoire père du répertoire courant !

# Importation de la librairie iads
import iads as iads

# importation de Classifiers
from iads import Classifiers as classif

# importation de utils
from iads import utils as ut


start = time.time()
un_KNN=classif.ClassifierKNN(2,1)
un_KNN.train(X2D,Y2D)
ut.plot_frontiere(X2D, Y2D,un_KNN)
ut.plot2DSet(X2D, Y2D)
print("Accuracy: ",un_KNN.accuracy(X2D,Y2D))
print(time.time() - start)