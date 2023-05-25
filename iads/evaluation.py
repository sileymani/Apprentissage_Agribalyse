# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt

def crossval(X, Y, n_iterations, iteration):
    n = len(X)
    test_start = (iteration * n) // n_iterations
    test_end = ((iteration + 1) * n) // n_iterations
    Xtest = X[test_start:test_end]
    Ytest = Y[test_start:test_end]
    Xapp = np.concatenate([X[:test_start], X[test_end:]])
    Yapp = np.concatenate([Y[:test_start], Y[test_end:]])
    return Xapp, Yapp, Xtest, Ytest

# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n_iterations, iteration):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    Y = Y[indices]
    X = X[indices]
    test_start = (iteration * n) // n_iterations
    test_end = ((iteration + 1) * n) // n_iterations
    Xtest, Ytest = [], []
    for c in np.unique(Y):
        indices_c = np.where(Y == c)[0]
        n_c = len(indices_c)
        test_start_c = (iteration * n_c) // n_iterations
        test_end_c = ((iteration + 1) * n_c) // n_iterations
        Xtest_c = X[indices_c[test_start_c:test_end_c]]
        Ytest_c = Y[indices_c[test_start_c:test_end_c]]
        Xtest.append(Xtest_c)
        Ytest.append(Ytest_c)
    Xtest = np.concatenate(Xtest)
    Ytest = np.concatenate(Ytest)
    Xapp = np.concatenate([X[:test_start], X[test_end:]])
    Yapp = np.concatenate([Y[:test_start], Y[test_end:]])
    return Xapp, Yapp, Xtest, Ytest

import copy

def validation_croisee(C, X,Y, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    perf = []
    for i in range(nb_iter):
        newC = copy.deepcopy(C)
        Xapp,Yapp,Xtest,Ytest = crossval_strat(X, Y, nb_iter, i)
        newC.train(Xapp, Yapp)
        acc_i=newC.accuracy(Xtest,Ytest)
        perf.append(acc_i)
        print(i,": taille app.= ",Yapp.shape[0],"taille test= ",Ytest.shape[0],"Accuracy:",acc_i)
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne=sum(L)/len(L)

    ecart_type=0
    for pref in L:
        ecart_type=ecart_type+((pref-moyenne)*(pref-moyenne))

    return (moyenne,math.sqrt(ecart_type/len(L)))

def validation_croisee_size(C, X, Y, nb_iter, sizes):
    """
        Cette fonction nous fait la validation croisé en fonction de sous ensemble de l'ensemble d'apprentissage.
        Classifieur * tuple[array, array] * int * list[int] -> tuple[list[float], float, float]
    """
    perf = []
    for train_size in sizes:
        acc_iter = []
        for i in range(nb_iter):
            newC = copy.deepcopy(C)
            Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
            # Réduire la taille de l'ensemble d'apprentissage
            Xapp = Xapp[:train_size]
            Yapp = Yapp[:train_size]
            newC.train(Xapp, Yapp)
            acc_i = newC.accuracy(Xtest, Ytest)
            acc_iter.append(acc_i)
            #print(f"Iteration {i}: taille app. = {train_size}, taille test = {Ytest.shape[0]}, Accuracy: {acc_i}")
        perf.append(acc_iter)
    
    # Calcul des moyennes et des écarts types des performances
    perf_moy = [np.mean(acc_iter) for acc_iter in perf]
    perf_sd = [np.std(acc_iter) for acc_iter in perf]
    
    return perf, perf_moy, perf_sd   

def plot_performance(train_sizes, perf_moy_list, perf_sd_list, labels):
    """
        list[int] * list[list[float]] * list[list[float]] * list[str] -> None
    """ 
    
    for perf_moy, perf_sd, label in zip(perf_moy_list, perf_sd_list, labels):
        plt.errorbar(train_sizes, perf_moy, yerr=perf_sd, marker='o', label=label)
    
    plt.xlabel('Taille de l\'ensemble d\'apprentissage')
    plt.ylabel('Accuracy')
    plt.title('Accuracy en fonction de la quantité de données')
    plt.legend()
    plt.show()


# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
