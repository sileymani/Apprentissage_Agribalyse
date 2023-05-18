# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import scipy.cluster.hierarchy
from scipy.spatial.distance import cdist

# ------------------------ 
def normalisation(df):
    """
    Normaliser chaque colonne d'un DataFrame entre 0 et 1.
    """
    return (df - df.min()) / (df.max() - df.min())

import math

def dist_euclidienne(vecteur_1, vecteur_2):
    """
    Calcule la distance euclidienne entre deux vecteurs.
    """
    # Convertir les vecteurs en tableaux numpy
    vecteur_1 = np.array(vecteur_1)
    vecteur_2 = np.array(vecteur_2)
    
    # Calculer la distance euclidienne
    distance = np.sqrt(np.sum(np.power(vecteur_1 - vecteur_2, 2)))
    
    return distance

def centroide(data):
    """
    Calcule le centre de gravité (centroïde) d'un dataframe ou un np.array.
    """
    # Convertir le dataframe ou np.array en tableaux numpy
    data = np.array(data)
    
    # Calculer la moyenne pour chaque colonne
    moyenne = np.mean(data, axis=0)
    
    return moyenne

def dist_centroides(data1, data2):
    """
    Calcule la distance euclidienne entre deux groupes de vecteurs.
    """
    # Calculer les centroïdes pour chaque groupe de vecteurs
    centroide1 = np.mean(data1, axis=0)
    centroide2 = np.mean(data2, axis=0)
    
    # Calculer la distance euclidienne entre les deux centroïdes
    distance = np.linalg.norm(centroide1 - centroide2)
    
    return distance

def initialise_CHA(DF):
    partition = {}
    for i in range(len(DF)):
        partition[i] = [i]
    return partition

def fusionne(df, partition, verbose=False):
    dist_min = +np.inf
    k1_dist_min, k2_dist_min = -1,-1
    p_new = dict(partition)
    for k1,v1 in partition.items():
        for k2,v2 in partition.items():
            if k1!=k2:
                dist= dist_centroides(df.iloc[v1], df.iloc[v2])
                if dist < dist_min:
                    dist_min = dist
                    k1_dist_min, k2_dist_min = k1, k2
    if k1_dist_min != -1:
        p_new.pop(k1_dist_min)
        p_new.pop(k2_dist_min)
        p_new[max(partition)+1] = [*partition[k1_dist_min], *partition[k2_dist_min]]
    if verbose and k1_dist_min !=-1:
        print("Distance mininimale trouvée entre ["+str(k1_dist_min) +"," +str(k2_dist_min) +"]  = "+str(dist_min))
    return p_new, k1_dist_min, k2_dist_min, dist_min

def CHA_centroid(df,verbose=False,dendrogramme=False):
    result = []
    partition = initialise_CHA(df)
    for o in range(len(df)):
        partition,k1, k2, distance = fusionne(df, partition,verbose)
        if k1!=-1 and k2 !=-1:
            result.append([k1, k2, distance, len(partition[max(partition.keys())])])
        
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(result, leaf_font_size=24.,)

        # Affichage du résultat obtenu:
        plt.show()
    return result

def dist_linkage(linkage, arr1, arr2):
    r = cdist(arr1,arr2, 'euclidean')
    if linkage == 'complete':
        return np.max(r)
    if linkage == 'simple':
        return np.min(r)
    if linkage == 'average':
        return np.mean(r)
    
def fusionne_linkage(df, linkage,partition, verbose=False):
    dist_min = +np.inf
    k1_dist_min, k2_dist_min = -1,-1
    p_new = dict(partition)
    for k1,v1 in partition.items():
        for k2,v2 in partition.items():
            if k1!=k2:
                dist= dist_linkage(linkage,df.iloc[v1], df.iloc[v2])
                if dist < dist_min:
                    dist_min = dist
                    k1_dist_min, k2_dist_min = k1, k2
    if k1_dist_min != -1:
        p_new.pop(k1_dist_min)
        p_new.pop(k2_dist_min)
        p_new[max(partition)+1] = [*partition[k1_dist_min], *partition[k2_dist_min]]
    if verbose and k1_dist_min !=-1:
        print("Distance mininimale trouvée entre ["+str(k1_dist_min) +"," +str(k2_dist_min) +"]  = "+str(dist_min))
    return p_new, k1_dist_min, k2_dist_min, dist_min

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    ############################ A COMPLETER
    
    if linkage=='centroid':
        return CHA_centroid(DF,verbose,dendrogramme)
    
    result = []
    partition = initialise_CHA(DF)
    for o in range(len(DF)):
        partition,k1, k2, distance = fusionne_linkage(DF,linkage,partition,verbose)
        if k1 !=-1 and k2 != -1:
            result.append([k1, k2, distance, len(partition[max(partition.keys())])])
        
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(result, leaf_font_size=24.,)

        # Affichage du résultat obtenu:
        plt.show()
    return result

def CHA_complet(DF,verbose=False,dendrogramme=False):
    return CHA(DF,'complete',verbose,dendrogramme)

def CHA_simple(DF,verbose=False,dendrogramme=False):
    return CHA(DF,'simple',verbose,dendrogramme)

def CHA_average(DF,verbose=False,dendrogramme=False):
    return CHA(DF,'average',verbose,dendrogramme)

