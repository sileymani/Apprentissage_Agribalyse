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
from scipy.spatial.distance import euclidean
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
    partition = {i: [i] for i in range(len(DF))}
    return partition

def fusionne(df, partition, verbose=False):
    dist_min = np.inf
    k1_dist_min, k2_dist_min = -1, -1
    p_new = dict(partition)
    
    for k1 in partition:
        for k2 in partition:
            if k1 < k2:
                dist = dist_centroides(df.iloc[partition[k1]], df.iloc[partition[k2]])
                if dist < dist_min:
                    dist_min = dist
                    k1_dist_min, k2_dist_min = k1, k2
    
    if k1_dist_min != -1:
        p_new.pop(k1_dist_min)
        p_new.pop(k2_dist_min)
        p_new[max(partition)+1] = partition[k1_dist_min] + partition[k2_dist_min]
    
    if verbose and k1_dist_min != -1:
        print("Distance minimale trouvée entre [" + str(k1_dist_min) + "," + str(k2_dist_min) + "] = " + str(dist_min))
    
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

def inertie_cluster(Ens):
    if isinstance(Ens, pd.DataFrame):
        Ens = Ens.values
    CK = np.mean(Ens, axis=0) # Centroids
    distances = np.sum((Ens-CK)**2, axis=1)
    inertia = np.sum(distances)
    return inertia

def init_kmeans(K,Ens):
    df_Ens=pd.DataFrame(Ens)
    return np.array(df_Ens.sample(n=K))

def plus_proche(Exe,Centres):
    if isinstance(Exe, pd.DataFrame):
        Exe = Exe.values
    c_proche = 0
    d = np.sum((Exe-Centres[0])**2)
    for i in range(len(Centres)):
        temp = np.sum((Exe - Centres[i])**2)
        if temp < d:
            d = temp
            c_proche = i
    return c_proche

def affecte_cluster(Base,Centres):
    dict_centre={i:[] for i in range(0,len(Centres))}
    for j in range (0,len(Base)):
        dict_centre[plus_proche(np.array(Base)[j],Centres)].append(j)
        
    return dict_centre

def nouveaux_centroides(Base,U):
    Base_numpy=np.array(Base) #pour pouvoir utiliser np.mean
    result=[]
    for k,elems in U.items():
        result.append(np.mean([Base_numpy[i] for i in elems],axis=0))
    
    return np.array(result)

def inertie_globale(Base, U):
    Base_numpy=np.array(Base)
    return sum([inertie_cluster([Base_numpy[i] for i in valeur]) for valeur in U.values()])
    
def kmoyennes(K, Base, epsilon, iter_max,affichage=True):
    Centres=init_kmeans(K,Base)
    U=affecte_cluster(Base,Centres)
    inertie_nouv=inertie_globale(Base,U)
    if affichage:
        print("Iteration : ",1," Inertie : ",inertie_nouv," Difference : ",inertie_nouv-epsilon-1)
    for i in range(1,iter_max):
        inertie_ancien=inertie_nouv
        #Recalcul de Centres et U
        Centres=nouveaux_centroides(Base,U)
        U=affecte_cluster(Base,Centres)
        inertie_nouv=inertie_globale(Base,U)
        diff=inertie_ancien-inertie_nouv
        if affichage:
            print("Iteration : ",i+1," Inertie : ",inertie_nouv," Difference : ",diff)
        if (diff < epsilon):
            break
        
    return Centres,U

def index_Dunn(Base,Centres,U):
    
    dist_matrix = np.zeros((len(Base), len(Base)))
    for i in range(len(Base)):
        for j in range(i + 1, len(Base)):
            distance = np.sqrt(np.sum((Base.iloc[i] - Base.iloc[j])**2))
            dist_matrix[i][j] = distance
            dist_matrix[j][i] = distance

    # Calculer la distance la plus courte entre tous les points internes des grappes,
    diameters = []
    for group in U.values():
        if len(group) > 1:
            group_matrix = dist_matrix[group][:, group]
            diameters.append(np.max(group_matrix))
        else:
            diameters.append(0)
    max_diameter = max(diameters)
    
    
    L=[]
    for i in range(len(Centres)):
        for j in range(len(Centres)):
            if(i!=j):
                L.append((i,j))
    list_dist=[clust.dist_centroides(Centres[i],Centres[j])for (i,j) in L]
    
    
    return max(diameters)/min(list_dist)

def index_beni(Base,Centres,U):
   
    dispersions = []
    for group in U.values():
        group_center = np.mean(Base.iloc[group], axis=0)
        group_dispersion = np.mean([np.sqrt(np.sum((Base.iloc[i] - group_center)**2)) for i in group])
        dispersions.append(group_dispersion)

    spacings = []
    for i, center_i in enumerate(Centres):
        for j, center_j in enumerate(Centres):
            if i < j:
                spacing = np.sqrt(np.sum((center_i - center_j)**2))
                spacings.append(spacing)
    avg_spacing = np.mean(spacings)

    
    vb_index = sum(dispersions) / avg_spacing

    return vb_index 

def clustering_info(resultat,DF,k):
    # Récupérez les indices des exemples de chaque cluster
    indices_clusters = [set(cluster[:2]) for cluster in resultat[-k:]]  # Remplacez k par le nombre de clusters souhaité

    # Créez un dictionnaire pour stocker les aliments par cluster
    clusters_aliments = {}

    # Parcourez les exemples du dataframe
    for i, row in DF.iterrows():
        indice_exemple = i
        aliment = row['Nom du Produit en Français']
        
        # Vérifiez à quel cluster appartient l'exemple
        for cluster, indices in enumerate(indices_clusters):
            if indice_exemple in indices:
                # Ajoutez l'aliment au cluster correspondant dans le dictionnaire
                if cluster not in clusters_aliments:
                    clusters_aliments[cluster] = []
                clusters_aliments[cluster].append(aliment)
                break

    clusters_aliments = dict(sorted(clusters_aliments.items(), key=lambda x: x[0]))

    # Affichez les aliments de chaque cluster
    for cluster, aliments in clusters_aliments.items():
        print(f"Cluster {cluster}:")
        for aliment in aliments:
            print(aliment)
        print()
        
        
def affiche_resultat(Base,Centres,Affect):

    X=[]
    Y=[]
    Liste=[]
    Base2 = Base
    Base = np.asarray(Base)
    dim=len(Base[0])
    colors=['b', 'g', 'y','c', 'm']
    t=0
    for i,L in Affect.items():
        
        for i in (Base[L]):
            
            X.append(i[0])
            Y.append(i[1])
            
            
    
        plt.scatter(X, Y,c=colors[t])
        t+=1     
        X=[]
        Y=[]
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
        
    plt.title('Nuage de points avec Matplotlib')
    plt.xlabel(Base2.columns[0])
    plt.ylabel(Base2.columns[1])
    plt.savefig("fig2D.png")
    plt.show()
    
    
def cluster_nocif(resultat, DF, attribute1, attribute2):
    # Récupérez les indices des exemples de chaque cluster
    indices_clusters = [list(cluster[:2]) for cluster in resultat]

    # Calculer la mesure de nocivité combinée pour chaque cluster
    cluster_nocivity = []
    for indices in indices_clusters:
        cluster_data = DF.loc[indices, [attribute1, attribute2]]
        cluster_nocivity.append(cluster_data.mean(axis=0).sum())

    # Trouver l'indice du cluster avec la valeur maximale
    max_index = cluster_nocivity.index(max(cluster_nocivity))

    # Récupérer les aliments du cluster le plus nocif
    max_cluster_indices = indices_clusters[max_index]
    max_cluster_aliments = DF.loc[max_cluster_indices, 'Nom du Produit en Français']

    return max_cluster_aliments
