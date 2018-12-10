# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:09:11 2017

@author: cbothore
"""


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter



def naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node (from empty), value is a list of attribute values. Here 
       only 1 value in the list.
     """
    predicted_values={}
    for n in empty:
        nbrs_attr_values=[] 
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values
    
 
def evaluation_accuracy(groundtruth, pred):
    """    Compute the accuracy of your model.

     The accuracy is the proportion of true results in the completed nodes.

    Parameters
    ----------
    groundtruth :  : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.
    pred : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values. 

    Returns
    -------
    out : float
       Accuracy.
    """
    true_positive_prediction=0   
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction+=1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            true_positive_prediction += len([c for c in p_value if c in groundtruth[p_key]])          
        # no else, should not happen: train and test datasets are consistent
    return true_positive_prediction*100/len(pred)
   

# load the graph
G = nx.read_gexf("/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumLinkedin.gexf")
print("Nb of users in our graph: %d" % len(G))

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes
# Some nodes may have 1 attribute 'location'
# Some nodes may have 1 or more 'colleges' or 'employers', so we
# use dictionaries to store the attributes
college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)

# here are the empty nodes for whom your challenge is to find the profiles
empty_nodes=[]
with open('/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
print("Your mission, find attributes to %d users with empty profile" % len(empty_nodes))

# here is the truth
groundtruth_employer={}
with open('/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
    
groundtruth_college={}
with open('/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumCollege.pickle', 'rb') as handle:
    groundtruth_college = pickle.load(handle)
    
groundtruth_location={}
with open('/Users/mathilde/Documents/École/IMT_Atlantique/S2/data_science/Challenge2/mediumLocation.pickle', 'rb') as handle:
    groundtruth_location = pickle.load(handle)
    
# --------------------- Baseline method -------------------------------------#
# Try a naive method to predict attribute
# This will be a baseline method for you, i.e. you will compare your performance
# with this method
# Let's try with the attribute 'employer'


loc_predictions=naive_method(G, empty_nodes, location)
result=evaluation_accuracy(groundtruth_location,loc_predictions)
print("%f%% of the predictions are true" % result)

# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes

###Data understanding

#stat de base

# print("Nb of users with one or more attribute college: %d" % len(college))
# print("Nb of users with one or more attribute location: %d" % len(location))
# print("Nb of users with one or more attribute employer: %d" % len(employer))
# 
# print("Nbre d'utilisateurs:",len(nx.nodes(G)))
# print("Nbre de relations:",len(nx.edges(G)))
# 
# plt.figure("Histogramme des degrés")
# plt.title("Histogramme des degrés")
# plt.hist([d for n,d in G.degree],bins ='auto')
# plt.xlabel("degrés")
# plt.ylabel("Nbr d'utilisateurs")
# plt.show()
# 


#création de 3 dictionnaires associants les attributs au nombre de personnes qui les possèdent

def stat_attributs(attribut):
    '''
    va rendre un dictionnaire qui à chaque attribut (college, location, employer) renvoit le nombre de personnes le possedant
    input:dictionnaire associant à chaque user son attribut (ou pas)
    output:dictionnaire
    '''
    final = {}
    for x in attribut :
        for element in attribut[x] :
            if not element in final:
                final[element]=1
            else:
                final[element]+=1
    return final
    
print('Nb of different colleges : ' + str(len(stat_attributs(college))))
print('Nb of different locations : ' + str(len(stat_attributs(location))))
print('Nb of different employers : ' + str(len(stat_attributs(employer))))    


# Probabilité d'avoir un attribut commun sachant que les utilisateurs sont en relation (apparement amélioré par Alexandre)
def proba_attribut_commun_sachant_edge():
    loc_nbrRelations = 0
    emp_nbrRelations = 0
    col_nbrRelations = 0
    
    loc_EnCommun = 0
    emp_EnCommun = 0
    col_EnCommun = 0
    
    for user1,user2 in list(G.edges):
        # locations
        if user1 in attribut and user2 in attribut:
            loc_nbrRelations += 1
            if location[user1] == location[user2]:
                loc_EnCommun += 1
        # college
        if user1 in college and user2 in college:
            col_nbrRelations += 1
            if college[user1] == college[user2]:
                col_EnCommun += 1
        # employer
        if user1 in employer and user2 in employer:
            emp_nbrRelations += 1
            jobsEnCommun = 0 # on calcule la probabilité d'avoir le même emploi à cet instant
            for job in employer[user1]:
                if job in employer[user2]:
                    jobsEnCommun += 1
            emp_EnCommun += jobsEnCommun/(len(employer[user1])*len(employer[user2]))
    print("Probabilité d'habiter au même endroit sachant qu'il y a une relation",loc_EnCommun/loc_nbrRelations)
    print("Probabilité de travailler au même endroit sachant qu'il y a une relation",emp_EnCommun/emp_nbrRelations)
    print("Probabilité d'avoir étudié au même endroit sachant qu'il y a une relation",col_EnCommun/col_nbrRelations)

#calcul d'HOMOPHILY : quelles sont les attributs qui rapprochent les gens ? formule pas très importante pour l'ensemble, plutôt pour des communautés. Le choix d'être connecté par LinkedIn est il fait par le lien de college, adresse, employer ?

def coef_adjacence(user1,user2):
    if (user1,user2) in G.edges():
        return 1
    else:
        return 0

def nb_arretes(attribut):
    m = 0
    for user1 in attribut:
        for user2 in attribut:
            if coef_adjacence(user1,user2) == 1:
                m+=1
    return m

def test_similarity(user1,user2,attribut):
    for lieu in attribut[user1]:
        return lieu in attribut[user2]

def modularity(attribut): #on se place dans les sous-graphes où les gens ont renseigné leur attribut
    m = nb_arretes(attribut) #nombre d'arrètes dont les sommets ont renseigné attributs
    final = 0
    for user1 in attribut:
        k1 = nx.degree(G)[user1]
        for user2 in attribut:
            k2 = nx.degree(G)[user2]
            adjacence = coef_adjacence(user1,user2)
            if test_similarity(user1,user2,attribut):
                final += (adjacence - (k1*k2)/(2*m))/(2*m)
    return final

# for x in (college,employer,location):
#     print('lhomophily par lien de pour ceux qui ont renseigné leur attribut : ' + str(modularity(x)))
            
#Calcul de la prof

def homophily(attribut):
    similar_neighbors=0
    total_number_neighbors=0 # to verify the number of edges ;-)!!!
    for n in G.nodes():
        for nbr in G.neighbors(n):
            total_number_neighbors+=1
            if n in attribut and nbr in attribut:
                if len([val for val in attribut[n] if val in attribut[nbr]]) > 0:
                    similar_neighbors+=1
    homophily=similar_neighbors/total_number_neighbors
    print("Homophily ('attribut' attribute), i.e. total proportion of neighbors sharing attributes: %f" % homophily)
    print('Is our number of edges (=%d) similar to networkX property (=%d)? No? Normal! why?' % (total_number_neighbors, len(G.edges())))

#détection de COMMUNAUTÉS : methode de Louvain
import community

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure

#first compute the best partition
partition = community.best_partition(G)

# #drawing
# size = float(len(set(partition.values())))
# pos = nx.spring_layout(G)
# count = 0.
# for com in set(partition.values()) :
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys()
#                                 if partition[nodes] == com]
#     nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
#                                 node_color = str(count / size))
# nx.draw_networkx_edges(G,pos, alpha=0.5)
# plt.show()

###Data manipulation

global nbr_iteration
nbr_iteration = 0

def attr_by_neigbours(graph, empty, attr,poids):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    complexité : O(len(empty_nodes)*len(neighbors)*?)
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.
    poids : dict
       A dict of attributes, which are a dict of value, which is the weigh the attribute has on the user. 
       key is a node (from empty), value is a dict of attribute values of nb. Here the nb are inferior than 1.
    Returns
    -------
    poids : dict 
       A dict of attributes, which are a dict of value, which is the weigh the attribute has on the user. 
       key is a node (from empty), value is a dict of attribute values of nb. Here the nb are inferior than 1.
     """    
    global nbr_iteration
    nbr_iteration += 1
    for n in empty:
        for nbr in graph.neighbors(n):
            if nbr in poids:
                for val in poids[nbr]:
                    if n in poids:
                        if val in poids[n]:
                            poids[n][val] = poids[n][val] + poids[nbr][val]/(G.degree(n)*nbr_iteration)
                        else:
                            poids[n][val] = poids[nbr][val]/(G.degree(n)*nbr_iteration)
                    else:
                        poids[n]={}
                        poids[n][val] = poids[nbr][val]/(G.degree(n)*nbr_iteration)
    return poids

def remplir_poids(graph,empty,attr):
    #initialisation
    poids = {}
    for u in attr:
        poids[u]={}
        for val in attr[u]:
            poids[u][val]=1/len(attr[u])
    pile = [] #à remplir
    for u in empty:
        pile.append(u)
    #while len(pile) != 0:
    for x in range(3):
        print(x)
        poids = attr_by_neigbours(graph, empty, attr, poids)
        for u in poids:
            if u in pile:
                pile.remove(u)
    return poids
    
def choix_attribut(graph,empty,attr):
    poids = remplir_poids(graph,empty,attr)
    prediction = {}
    for u in empty :
        if u in poids:
            prediction[u]=[]
            for val in poids[u]:
                if poids[u][val]>0.00001:
                    prediction[u].append(val)
    return prediction

college_predictions = choix_attribut(G,empty_nodes,college)
result=evaluation_accuracy(groundtruth_college,college_predictions)
print("%f%% of the predictions are true" % result)
print(len(college_predictions)/len(empty_nodes))

loc_predictions = choix_attribut(G,empty_nodes,location)
result=evaluation_accuracy(groundtruth_location,loc_predictions)
print("%f%% of the predictions are true" % result)

employer_predictions = choix_attribut(G,empty_nodes,employer)
result=evaluation_accuracy(groundtruth_employer,employer_predictions)
print("%f%% of the predictions are true" % result)

#par communauté:

def attr_by_community(G,empty):
    partition = community.best_partition(G)
    poids_college = remplir_poids(graph,empty,college)
    poids_location = remplir_poids(graph,empty,location)
    poids_college = remplir_poids(graph,empty,college)