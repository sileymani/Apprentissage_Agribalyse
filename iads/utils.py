# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


# ------------------------ REPRENDRE ICI LES FONCTIONS SUIVANTES DU TME 2:
#  genere_dataset_uniform:
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    d1 = np.random.uniform(binf, bsup, (2 * n, p))
    d2 = np.asarray([-1 for i in range(0, n)] + [1 for i in range(0, n)])
    np.random.shuffle(d2)
    return (d1, d2)
    raise NotImplementedError("Please Implement this method")


# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    n_pos = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    n_neg = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    n_tout = np.concatenate([n_pos, n_neg])
    data_label = np.asarray([-1 for i in range(0, nb_points)] + [1 for i in range(0, nb_points)])
    n_tout, data_label = shuffle(n_tout, data_label)
    return (n_tout, data_label)
    raise NotImplementedError("Please Implement this method")


# plot2DSet:
def plot2DSet(desc, labels):
    desc_negatifs = desc[labels == -1]
    desc_positifs = desc[labels == +1]
    plt.scatter(desc_negatifs[:, 0], desc_negatifs[:, 1], marker='o', color="red")  # 'o' rouge pour la classe -1
    plt.scatter(desc_positifs[:, 0], desc_positifs[:, 1], marker='x', color="blue")  # 'x' bleu pour la classe +1


#  plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    # tracer des frontieres
    #  colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid, x2grid, res, colors=["darksalmon", "skyblue"], levels=[-1000, 0, 1000])

