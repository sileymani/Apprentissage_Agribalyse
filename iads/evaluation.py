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

# def crossval_strat(X, Y, n_iterations, iteration):
#     X1 = X[Y==1]
#     Y1 = Y[Y==1]
#     X_1 = X[Y==-1]
#     Y_1 = Y[Y==-1]
#     Xapp1, Yapp1, Xtest1, Ytest1 = crossval(list(X1), list(Y1), n_iterations, iteration)
#     Xapp_1, Yapp_1, Xtest_1, Ytest_1 = crossval(list(X_1), list(Y_1), n_iterations, iteration)
#     Xapp = Xapp_1 +  Xapp1
#     Yapp = Yapp_1 + Yapp1
#     Xtest = Xtest_1 + Xtest1
#     Ytest = Ytest_1 + Ytest1
#     return np.asarray(Xapp), np.asarray(Yapp), np.asarray(Xtest), np.asarray(Ytest)

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


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne = sum(L)/len(L)
    
    ecart_type=0
    for pref in L:
        ecart_type=ecart_type+((pref-moyenne)*(pref-moyenne))
        
    return (moyenne,math.sqrt(ecart_type/len(L)))
    raise NotImplementedError("Vous devez implémenter cette fonction !")   

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS 
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

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
