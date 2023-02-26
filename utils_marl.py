import numpy as np
from scipy.io import loadmat
import numpy as np
import networkx as nx
import glob
import string
import operator
import copy
import sys
import pandas as pd
import pickle
import dill
import itertools
from pathlib import Path
import os

from collections import deque

#Plot Traffic
import matplotlib.pyplot as plt

from itertools import islice
from scipy.io import loadmat
import numpy as np
import math
import copy
import networkx as nx

import tensorflow as tf
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import pickle
import sys

from itertools import chain
import random
import time
import datetime as dt

#from *file* import *class/classes*
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from statics import *


def reading_demands_creating_TM(matrice):

    diz_total = {}
    #diz_total_day = {}

    #[sum of all plot elements / somma di tutti gli elementi plot]
    for ii in range(len(matrice)):
        reteName = matrice[ii]

        with open(reteName, "r") as a:
            data = a.readlines()

        check = 0
        nodes = {}
        n = 1

        for line in data:
            if "NODES" in line:
                check = 1
            if check == 1:
                words = line.split()
                if words[0] == ")":
                    check = 0
                if words[0] != "NODES" and check == 1:
                    nodes[words[0]] = str(n)
                    n += 1

        check = 0
        l = 1

        check = 0
        d = 0

        demand = []

        for line in data:

            if "DEMANDS" in line:
                check = 1
            if check == 1:
                words = line.split()
                if words[0] == ")":
                    check = 0

                if words[0] != "DEMANDS" and check == 1:
                    #it's possible to change the unity of measure; we are using now Mbs /
                    #possibile cambiare l'unità di misura per ora stiamo in Mbs

                    demand_list = [d, int(nodes[words[2]])-1, int(nodes[words[3]])-1, float(words[6])*1e6]

                    d += 1

                    demand.append(demand_list)

        diz_total[ii] = np.array(demand, dtype = object)

    return diz_total


def createTM(matrix_day):

    TM = np.zeros((number_nodes, number_nodes),dtype = 'object')

    for i in range(matrix_day.shape[0]):
        src = matrix_day[i,1]
        dst = matrix_day[i,2]
        val = matrix_day[i,3]
        #print(src,dst,val,i)
        TM[src][dst] = val

    return TM

def vec_ratio(data):
    summ = np.sum(data[:,3])
    ratio = data[:,3]/summ

    return ratio

def Extract_maxIndex(dizionario_blu):
    max_list = [np.sum(dizionario_blu[i]) for i in range(len(dizionario_blu))]

    index, value = max(enumerate(max_list), key=operator.itemgetter(1))
    #print("Max Index for Blu Matrix: " + str(index))
    return index

def calcolaIGPpaths(L, N, A, linkRete, ECMP,Graph):

    IGP_paths = [ [[]]*N*(N-1) for i in range(L)]


    cammini = [ [[]]*N for i in range(N)]
    k = 0

    if ECMP == 0:
        #matrix of the Next-Hop
        NH = np.zeros((N,N), dtype = int)

        for s in range(N):
            path = nx.single_source_shortest_path(Graph, s)

            for d in range(N):
                if s!= d:
                    #print(s,d)
                    #put the second node of the path within the dictionary
                    NH[s, d] = path[d][1]

    for s in range(N):
        for d in range(N):
            if s != d:
                if ECMP == 1:
                    print("No")
                else:

                    path = costruisciPathDaMatriceNH(NH, s, d, N)
                    cammini[s][d] = [int(i) for i in list(path)]
                    supp = costruisciSupporto(cammini[s][d], linkRete, L)
                    #print(supp)
                    for h in range(len(supp)):
                    #    print(supp[h])
                        IGP_paths[int(h)][k] = supp[h]

                    k = k + 1

    return IGP_paths, cammini, NH


def costruisciPathDaMatriceNH(NH, s, d, N):
    path = np.zeros(N)
    path[0] = s;
    nh = NH[s, d]

    path[1] = nh

    k = 2

    while nh != d:

        nh = NH[int(path[k - 1]), d]

        path[k] = nh
        k = k + 1


    path = path[0:k]
    #print(s,d)
    #print(path)

    return path

def costruisciSupporto(shortestPaths, linkRete, L):

    #print(shortestPaths)
    #print()
    supporto = np.zeros((L, 1))
    path = shortestPaths
    P = 1
    #print(path[0],path[1])
    for l in range(1,len(path)):
        link =linkRete[(linkRete[:,1]==path[l-1]) & (linkRete[:, 2] == path[l])][0][0]
        #supporto(link) = supporto(link) + (1 / P)
        link = int(link)
        supporto[link] = supporto[link] + 1
    #print(supporto)
    #print()
    #print()
    supporto = np.reshape(supporto,(1,-1))[0]
    return supporto


def calcolaSL_color1(N, R, IGP_paths, linkRete, flows):

    SL = [[[]]*N for i in range(N)]

    for s in range(N):
        for d in range(N):
            try:

                flusso = flows[(flows[:,1] == s) & (flows[:,2] == d) & (flows[:,3] == 1)][0]

            except:
                flusso = None
                #print(s,d)
                #print("except")
            if flusso is not None:

                SL[s][d] = [s,d]

                f = int(flusso[0])
                #print(f)
                R[:,f] = IGP_paths[:,f]

            #print()
            #print()

    return SL,R


def encoding(SL,cammini,action):
    #return the updated segment list give the new action we want to perform

    #print(action)
    sl_new = [action[0]]

    current = 0
    active_segment = 1

    while action[current] != action[-1]:

        #print(action[current],action[active_segment])
        #print()
        if action[current:active_segment+1] == cammini[action[current]][action[active_segment]]:

            active_segment += 1

            if active_segment == len(action):

                sl_new.append(action[-1])

                #print(sl_new)

                #SL[action[0]][action[-1]] = sl_new

                return sl_new

        else:

            sl_new.append(action[active_segment-1])
            #print(sl_new)
            current = active_segment -1


def startingPSID(SL, G, flows, N, linkRete):

    PSID = np.zeros((N*N,3),dtype = np.int64)
    Yb =  np.zeros((PSID.shape[0], 1))
    B = np.zeros((PSID.shape[0],flows.shape[0]))

    p = 0

    for i in range(N):
        for j in range(N):
            PSID[p,:] = [int(p),int(i),int(j)]
            p += 1


    for f in range(len(flows)):
        #extract segment list
        sl = SL[flows[f,1]][flows[f,2]]


        contatore = PSID[(PSID[:,1] == flows[f,1]) & (PSID[:,2] == flows[f,2])][0][0]
        B[contatore, f] = 1

        #Il traffico quando arriva al nodo di bordo il traffico viene contato rispetto al nodo di uscita
        Yb[contatore] = Yb[contatore] + flows[f, 4]

        #aggiornamento dell'Yb counter avviene in trasmissione (esempio [1,11,8,2] ricorda)

        for a in range(1,len(sl)):

            #contatore = PSID[(PSID[:,1] == sl[a-1]) & (PSID[:,2] == sl[a])][0][0]

            fid = flows[(flows[:,1] == sl[a-1]) & (flows[:,2] == sl[a])][0][0]

            #lista dei link attraversati dal flusso,
            tmp = linkRete[np.where(G[:,fid]> 0)[0],:]


            for l in  range(tmp.shape[0]):

                psid = PSID[(PSID[:,1] == tmp[l,2]) & (PSID[:,2] == sl[a])][0][0]

                #n segment routing puoi usare il multi path tra due nodi,
                #quindi potresti avere che un flusso venga diviso in duo o più parti.

                B[psid,f] = B[psid,f] + 1
                Yb[psid] = Yb[psid] + flows[f,4]*1


    return B,Yb, PSID,


def computeBandwidth(linkRete,R,flows):

    #moduli capacità
    moduli_cap = 1_000_000_000
    over_provisioning = 0.5

    #Calcolare il numero di moduli per link

    cap_link = np.ceil((R.dot(flows[:,-1])+1)/(over_provisioning*moduli_cap))

    linkRete[:,4] = cap_link*moduli_cap

    return linkRete


def create_Flows(num_color,TM,split_color,K,N):
    flows = np.zeros((K, 5), dtype = object)
    k=0

    for c in range(num_color):
        for i in range(N):
            for e in range(N):
                #print(k)
                if TM[i,e] != 0 :
                #if TM[i,e] >= 0 :

                    #Take the intensity of the Traffic Matrix
                    intensity = split_color[c] * TM[i,e]
                    #print(k)
                    flows[k, :] = [k, i, e, c+1, intensity]
                    k = k + 1

    return flows

def compute_LinkRete(max_matrix_blu):
    # parte duplicata in compute_allParameters
    TM = max_matrix_blu

    delay = np.zeros((number_nodes, number_nodes),dtype = 'object')
    bandwidth = np.zeros((number_nodes, number_nodes),dtype = 'object')

    #annots = loadmat(base + NETWORK_NAME + '_A.mat')
    #A = annots["A"]
    #Graph = nx.from_numpy_matrix(A)

    split_color = [1]
    ECMP = 0 #Equal-Cost Multi-path routing
    N = len(A)
    L = int(sum(sum(A)))
    num_color = 1

    #linkRete (ID_flow, src, dst, del, bandwidth )
    linkRete = np.zeros((L, 5),dtype = 'object')
    l = 0


    for i in range(N):
        for j in range(N):
            #if there is an edge between the 2 nodes
            if A[i,j] > 0:
                linkRete[l, :] = [l ,i, j, delay[i, j], bandwidth[i, j]]
                l = l + 1

    #It is necessary this part
    #A[A==0] = np.inf ???

    #-solo flussi diversi da 0
    K = len(TM[TM>0])*num_color
    #-consideriamo anche i flussi con 0
    #K = len(TM[TM>=0])*num_color

    flows = create_Flows(num_color,TM,split_color,K,N)

    K = len(flows)
    #print("Number of FLOWS: " + str(K))


    GG, cammini, NH = calcolaIGPpaths(L, N, A, linkRete, ECMP,Graph)

    R = np.zeros((L,K))

    G = np.array(GG)

    #Nel training non dobbiamo più ricalcolare calcolaSL_color1
    SL1, R = calcolaSL_color1(N, R, G, linkRete, flows)

    #print()
    #print(SL1)

    SL = SL1

    B, YbExp, PSID = startingPSID(SL, G, flows, N, linkRete)

    #aggiorniamo la capacità dei link in modo corretto

    #Stabilire dimensione modulo

    linkRete = computeBandwidth(linkRete,R,flows)

    return linkRete, G

def compute_allParameters(matrix_analyzed,linkRete):
    # parte duplicata in Compute_linkRete
    TM = matrix_analyzed

    #delay = np.zeros((number_nodes, number_nodes),dtype = 'object')
    #bandwidth = np.zeros((number_nodes, number_nodes),dtype = 'object')

    #annots = loadmat(base + NETWORK_NAME + '_A.mat')
    #A = annots["A"]
    #Graph = nx.from_numpy_matrix(A)

    split_color = [1]
    ECMP = 0 #Equal-Cost Multi-path routing
    N = len(A)
    L = int(sum(sum(A))) # total number of link in the network
    num_color = 1


    #It is necessary this part
    #A[A==0] = np.inf ???

    #-solo flussi diversi da 0
    K = len(TM[TM>0])*num_color
    #-consideriamo anche i flussi con 0
    #K = len(TM[TM>=0])*num_color

    flows = create_Flows(num_color,TM,split_color,K,N)

    K = len(flows)
    #print("Number of FLOWS: " + str(K))


    GG, cammini, NH = calcolaIGPpaths(L, N, A, linkRete, ECMP,Graph)

    R = np.zeros((L,K))

    G = np.array(GG)

    #Nel training non dobbiamo più ricalcolare calcolaSL_color1
    SL1, R = calcolaSL_color1(N, R, G, linkRete, flows)

    #print()
    #print(SL1)

    SL = SL1

    B, YbExp, PSID = startingPSID(SL, G, flows, N, linkRete)

    #aggiorniamo la capacità dei link in modo corretto

    #Stabilire dimensione modulo

    return flows,R,SL,YbExp,PSID,cammini,L,K

def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target,), k))

def customize_bandwidth(agents):

    print('in customize: ', agents)

    linkRete = pd.read_pickle(base + "LinkReteNEW.pkl")

    if NETWORK_NAME == 'germany':

        # general setting
        for i in range(linkRete.shape[0]):

            valBand = linkRete[i,-1]

            if valBand <= 2000:
                linkRete[i,-1] = 1000# for node 25
                #linkRete[i,-1] = 1800
            else:
                linkRete[i,-1] = 2500

        #setting for specific nodes
        for node in agents:
            if node == 49:
                # stava già bene così
                pass

            if node == 45:
                linkRete[160, -1] = 900
                linkRete[161, -1] = 1000

            if node == 43:
                linkRete[152,-1] = 500
                linkRete[149,-1] = 500

            if node == 39:
                linkRete[138, -1] = 300

            if node == 34:
                linkRete[120, -1] = 800

            if node == 32:
                linkRete[115, -1] = 700
                linkRete[116, -1] = 390

            if node == 28:
                linkRete[101, -1] = 900
                linkRete[99, -1] = 700


            if node == 13:
                linkRete[48, -1] = 2300
                linkRete[46, -1] = 1900

            if node == 11:
                linkRete[38, -1] = 700
                linkRete[39, -1] = 700
                linkRete[40, -1] = 700
                linkRete[41, -1] = 700

            if node == 48:
                linkRete[167, -1] = 700

    elif NETWORK_NAME == 'nobel':
        for i in range(linkRete.shape[0]):

            valBand = linkRete[i,-1]

            if valBand < 200:
                linkRete[i,-1] = 150
            else:
                linkRete[i,-1] = 200

            # general setting
            linkRete[:,-1] = 150
            linkRete[0,-1] = 250
            linkRete[6,-1] = 200
            linkRete[26,-1] = 180
            linkRete[7,-1] = 180

        for node in agents:

            if node == 15:
                linkRete[47, -1] = 80

            if node == 13:
                linkRete[41, -1] = 100
                linkRete[42, -1] = 100

            if node == 9:
                print('trovato nodo 9')
                linkRete[30, -1] = 40
                linkRete[32, -1] = 40

            if node == 8:
                print('trovato nodo 8')

                #nodo 16 (16?? non è 8? il 26 è da 8 a 1)
                linkRete[26, -1] = 180 # link 8 - 1
                linkRete[28, -1] = 80# link 8 - 9

            if node == 5:
                linkRete[19, -1] = 60
                linkRete[20, -1] = 60
                linkRete[21, -1] = 70

            if node == 0:
                print('trovato nodo 0')

                # nodo 0
                linkRete[0,-1] = 250
                linkRete[6,-1] = 200
                linkRete[26,-1] = 180
                linkRete[7,-1] = 180






    else:
        print('I can\'t recognize your network... ;) ' )
        exit()

    return linkRete



def fill_Encap(dict_encap,SL,cammini,action):
    #(id_agent,id_active_segment):SL
    #Possibile cambiamento per multi-agente

    #Single-agent
    active_segment = action[-1]

    dict_encap[active_segment] = encoding(SL,cammini,action)

    return dict_encap

def multi_fill_Encap(dict_encap_dict, SL, cammini, actions):
    # generalization of the fill encaps for multiple dict encap
    for id_node, dict_encap_id_node in dict_encap_dict.items(): # scanning the dict of dict
        if actions.get(id_node)!= None: # check if the node took an action
            #print('in multi fill encap - actions[id_node]', actions[id_node])
            dict_encap_dict[id_node] = fill_Encap(dict_encap_id_node, SL, cammini, actions[id_node]) #refill the dict encap with the encap of the node over the action

    return dict_encap_dict


def get_simple_path(complex_path):
    '''complex_path is a list of touples in the form: (node, active_segment)

        return the list of nodes'''

    simple_list = [first for (first, second) in complex_path]
    # we are taking just the first element of the touple composed by 2 elements: (first, second)
    return simple_list

def my_index(complex_path, element):
    '''complex_path is a list of touple of 2 elements (node, active_segment);
        element is a tuple representig an occurance in the complex_path

        the function return the index of the first occurance of element in the complex path
        otherwise return -1'''
    index=None

    try:
        index = complex_path.index(element)
    except:
        index = -1

    return index

def find_simple_cycle(path):
    '''this function find (not infilite) cycles in the path
        idea:
        check if the list of nodes and the set (without repetitions) of nodes have the same length
        '''

    path_nodes = get_simple_path(path)

    return not len(path_nodes) == len(set(path_nodes))

def find_worst_cycle(path):
    '''this function find worst (infinite) cycles in the path
        idea:
        check if the list of the path and the set (without repetitions) have the same length

        N.B. the path is a "complex path" i.e. list of (node, active_segment)

        example:
        flow: 217 from 13 to 9
        dict_encap_8 = {9 : [8, 16, 0, 9]}
        dict_encap_0 = {9 : [0, 16, 9]}
        '''
    return not len(path) == len(set(path))

def encapsulate(path, cammini, new_sl, index):
    # insert in the path the new sl in the index in witch the encapsulation must be build (ofcourse using the shortest path amond 2 consecutive active segment)
    simple_path = get_simple_path(path) # obtainin the path of only nodes (withous active segment)
    #print('simple_path: ', simple_path)
    end_encapsulate = len(simple_path[:index+1]) + simple_path[index+1:].index(new_sl[-1]) # fint the end of the encapsulation by lookn at nex occurance of the last active segment in the new sl
    #print('end_encapsulate:', end_encapsulate)
    # simple_path[index+1] significa che stimao cercando il punto in cui ricongiungere l'incapsulamento col path che avevamo già:
    # es: [(12, 1), (13, 1), (0, 13), (13, 15), (15, 1), (1, 1)], se cerchiamo il 13, non possiamo farlo su tutto il path ma dobbiamo
    # cercarli solo dopo aver incapsulato (0,13) e quindi index = 1 sarebbe errato

    # len(simple_path[:index+1]) poichè il pezzo successivo trova gli indici a aprtire da zero poiè opera su una sotto lista,
    # a questo indice va sommata la lunghezza della lista esclusa dalla sottolista

    path = path[:index] + path[end_encapsulate:] # taking the path from start to the start of the encapsulation on concatenate with the path from the end of the encapsulation to the end of the path itself

    #print('path orima di incapsulare: ', path)

    # general encapsulation... nothing new
    for id_sl in range(1,len(new_sl)):

        underlay_path = cammini[new_sl[id_sl-1]][new_sl[id_sl]]
        #print('encaps -  underlay_path\n', underlay_path)

        for id_under_path in range(len(underlay_path[:-1])):
            path.insert(index, (underlay_path[id_under_path], new_sl[id_sl]))

            #print('new total_underlay_path\n', path)

            index+=1

    return path

def agent_in_path(path, agent_action_list):
    # find if an agent in the tha path and have to incapsulate a new_sl
    for node, action, new_sl in agent_action_list: # iterating the triples of the dict_encap
        #print('mod: ', (node, action, new_sl))
        index = my_index(path, (node, action)) # try to find an index
        #print('index', index)
        if index != -1: # if found...
            return index, new_sl, (node, action) # return
    return -1, None, (None, None) # else return all negative



def multi_aggiornaPSID(SL, cammini, flows, N, linkRete, dict_encap_dict):
    '''This function is the MARL version of the aggiornaPSID function in order to implement it in a marl fashion'''
    # R : matrice di routing
    # SL : tutte le segment list iniziali

    # flows: matrice con 5 colonne; in ordine sono: id, sorgente, destinazione, "ignoralo", valore del flusso
    # i flussi sono tutti i possibili (che non ciclano)

    #cammini: tutti i cammini minimi per ogni coppia di nodi
    # matrice: cammini[sorgente][destinazione] -> lista dei nodi sul cammino minimo (ordinati)

    #link rete: matrice di tutti i link; con colonne; id del link, sorgente, destinazione, 'ignoralo', banda
    # sono in numero uguale agli archi del grafo

    # R: matrice di routing: tot number of link x number of flows

    #prefix segment ID
    PSID = np.zeros((N*N,3),dtype = np.int64) # tutte le coppie di nodi: numero, dove è instanziato il contatore (quello che sta ricevendo il pacchetto), sid per il quale di conta il traffico (ossia l'active segment):   andrebbe spostato furori
    # N * N perchè ogni nodo tinene conto dei pacchetti che provengono da gli altri nodi
    Yb =  np.zeros((PSID.shape[0], 1)) # contatori della quantità di flusso, uno per ogni PSID, indicano la quantità di flusso per la destinazione del PSID
    B = np.zeros((PSID.shape[0],flows.shape[0])) # 1 o 0: il flusso f è instradato verso il PSID p
    # NB B non ci serve a nulla ne qui ne fuori
    # matrice di PSID x flows:

    R = np.zeros((linkRete.shape[0],flows.shape[0])) # matrix of number of link times number of flows



    p = 0

    for i in range(N):
        for j in range(N):
            PSID[p,:] = [int(p),int(i),int(j)]
            p += 1

    #print('flowssssssssssssssssss', flows[(flows[:,1]==13) & (flows[:,2]==9)])
    #print('cammini 8 0 ', cammini[13][9])

    #dict_encap_dict = {0 : {13 : [0, 3, 13], 1 : [0, 13, 15, 1]}, 16 :  {5:  [16, 8, 0, 5], 14 : [16, 5, 12, 14]}}
    # prepare triples (agent, action and new_sl)
    agent_action_list = []
    for id_node, dict_encap in dict_encap_dict.items(): # iteranting over the esternal dict [DICT of dict]
        for action, encap in dict_encap.items(): # interating over internal dict [dict of DICT]
            if encap != [id_node, action]: # if an agent is not kepping the same initial incapsulation
                                            # example: {0, {9, [0, 9]}}
                agent_action_list.append((id_node, action, encap)) # add a triple of node, action and new_sl
    #print('lista di (agent,action):\n', agent_action_list)


    #flows = flows[flows[:,0]==193]


    for f in range(len(flows)): # for each flow
        #input('go\n\n\n\n\n')
        #print('\n\n\n\n\n\n\n\n')
        # aggiorniamo rispetto al nodo destinazione
        psid = PSID[(PSID[:,1] == flows[f,1]) & (PSID[:,2] == flows[f,2])][0][0] #-> identify the psid of the base flow

        #print('flow: ', flows[f, :])

        # PSID[:,1] == flows[f,1] -> per ogni PSID mi dice se è sorgente di quel flusso
        #   flows[f,1]: sorgente del flusso
        #   PSID[:,1]: sorgente del PSID            #with_cycle=find_cycle(total_underlay_path, applyed)

        # idem per la destinazione (seconda aprte)
        # PSID[(PSID[:,1] == flows[f,1]) & (PSID[:,2] == flows[f,2])] -> l'unico psid che è è quel flusso


        B[psid, f] = 1 # il flusso f viene instradato sul PSID p (psid)

        #Il traffico quando arriva al nodo di bordo viene contato rispetto al nodo di uscita (ancora non un pckt segment routing)
        Yb[psid] = Yb[psid] + flows[f, 4] # aggiorniamo il psid di quel PSID con il clarico del flusso che ci passa sopra
        #print('primo aggionramento PSID\n', PSID[psid])
        #aggiornamento dell'Yb counter avviene in trasmissione (esempio [1,11,8,2] ricorda)

        #print('!!! START COMPUTING THE TOTAL BIG PATH OF THE FLOW !!!!')

        #extract segment list
        #print('flow:\n', flows[f])
        sl = SL[flows[f,1]][flows[f,2]] #   ... segment list of the flow
        #print('segment list of the flow\n', sl)

        #print('!!! porcate !!! aggiungo qualcosa alla segment list')
        # sl+=[8, 9]

        #print('segment list con porcate:\n', sl)

        total_underlay_path = [ (flows[f,1], flows[f,2]) ] # (ingress node, active segment that is the destination of the flow)

        #print('start iterating over sl')
        for id_segment in range(1, len(sl)): # for each segment execpt the first
            #print('active segment: ', sl[id_segment])

            underlay_path = cammini[sl[id_segment -1]][sl[id_segment]] # obtain the shortest path from previous and nex segment

            #print('underlay_path[%a][%a] = %a' %(sl[id_segment -1], sl[id_segment], underlay_path))

            for id_under_path in range(1, len(underlay_path)): # iterating over the shortest path already found (except the first)
                total_underlay_path.append( (underlay_path[id_under_path], sl[id_segment]) ) # add the touple (node, active_segment)

        #print('total_underlay_path', total_underlay_path)


        with_cycle=False # total_underlay_path is just the composition of the underlay_path of the segment_list;
        # those underlay_paths are for sure without cycles and the composition as well
        incapsulate_index, new_sl, (node, action) = agent_in_path(total_underlay_path, agent_action_list) # try to find and encapsulation
        while incapsulate_index != -1: # until we have something to encapsulate
            #print('index: ', incapsulate_index, '       new_sl: ', new_sl)
            #print('modifico ********************************************************* ')
            #total_underlay_path = encapsulate(total_underlay_path, cammini, new_sl, incapsulate_index)
            total_underlay_path = encapsulate(total_underlay_path, cammini, new_sl, incapsulate_index) # encapsulate
            with_cycle=find_worst_cycle(total_underlay_path) # cheach if there are some worst cycles
            if with_cycle: # if there are
                #print('path with cycles:\n', total_underlay_path)
                #print('incaps index: ', incapsulate_index)
                #print('new_sl: ', new_sl)
                #print('(node, action): ', (node, action))
                #print('agent_action_list: ', agent_action_list)
                break # break
            else:
                incapsulate_index, new_sl, (node, action) = agent_in_path(total_underlay_path, agent_action_list) #find the next incapsulation
        #print('FINAL TOTAL UNDERLAY PATH:   ', total_underlay_path)

        if with_cycle:
            print_error('Abbiamo cicli! dobbiamo andare a reward negativa! come lo facciamo???????')
            #input()
            break # break the multi_aggiornaPSID
        # satrting updating

        for (node_s, act_seg_s), (node_e, act_seg_e) in zip(total_underlay_path[:-1], total_underlay_path[1:]): 
        #iterating over the a link i.e. 2 consecutive element in a complex path and so 2 touples of (node, active secment)

            #print((node_s, act_seg_s), (node_e, act_seg_e))

            id_link = linkRete[(linkRete[:, 1] == node_s) & (linkRete[:,2] == node_e)][0][0]

            #print('link rete in pos id_link\n', linkRete[id_link])

            R[id_link][f] = 1

            id_contatore = PSID[(PSID[:,1] == node_e) & (PSID[:,2] == act_seg_s)][0][0]
            #print('id_contatore: ', id_contatore)
            #print('psid in pos id_psid\n', PSID[id_contatore])

            B[id_contatore][f] = 1

            Yb[id_contatore] += flows[f,4]



    return B,Yb, PSID, R, with_cycle


def aggiornaPSID(SL, cammini, flows, N, linkRete, dict_encap, id_agent, R):

    PSID = np.zeros((N*N,3),dtype = np.int64)
    Yb =  np.zeros((PSID.shape[0], 1))
    B = np.zeros((PSID.shape[0],flows.shape[0]))

    p = 0

    for i in range(N):
        for j in range(N):
            PSID[p,:] = [int(p),int(i),int(j)]
            p += 1

    for f in range(len(flows)):
        #extract segment list
        sl = SL[flows[f,1]][flows[f,2]]
        #print("sl start")
        #print(sl)
        #print()
        #[1,8,2]

        contatore = PSID[(PSID[:,1] == flows[f,1]) & (PSID[:,2] == flows[f,2])][0][0]
        B[contatore, f] = 1 # da indagare Davide

        #Il traffico quando arriva al nodo di bordo viene contato rispetto al nodo di uscita (ancora non un pckt segment routing)
        Yb[contatore] = Yb[contatore] + flows[f, 4]

        #aggiornamento dell'Yb counter avviene in trasmissione (esempio [1,11,8,2] ricorda)

        for a in range(1,len(sl)):

            #cammino underlay
            tmp = cammini[sl[a-1]][sl[a]]

            for n in  range(len(tmp)-1):


                #il nodo "n" è l'agente?
                if tmp[n] != id_agent or (tmp[n] == id_agent and sl[a] not in dict_encap ):

                    #Classica azione update R , B Yb

                    id_link = linkRete[(linkRete[:,1]==tmp[n])&(linkRete[:,2]==tmp[n+1])][0][0]

                    R[id_link][f] = 1

                    id_contatore = PSID[(PSID[:,1] == tmp[n + 1]) & (PSID[:,2] == sl[a])][0][0]

                    B[id_contatore][f] = 1

                    Yb[id_contatore] += flows[f,4]

                else:

                    #Marco revisione fINISH

                    sl_new = dict_encap[sl[a]]

                    #print(sl_new)
                    #print()
                    #print()

                    for a_new in range(1,len(sl_new)):
                        #print(a_new,sl_new)

                        #cammino underlay
                        tmp = cammini[sl_new[a_new-1]][sl_new[a_new]]

                        for n in  range(len(tmp)-1):

                            id_link = linkRete[(linkRete[:,1]==tmp[n])&(linkRete[:,2]==tmp[n+1])][0][0]

                            R[id_link][f] = 1

                            id_contatore = PSID[(PSID[:,1] == tmp[n+1]) & (PSID[:,2] == sl_new[a_new])][0][0]

                            B[id_contatore][f] = 1

                            Yb[id_contatore] += flows[f,4]
                    break

    return B,Yb, PSID, R


def compute_newPath_length(old_path, action_selected):

    #print("Old Path: " + str(old_path))
    #print("New Path: " + str(action_selected))

    l_new = len(action_selected)
    l_old = len(old_path)

    diff = l_new - l_old

    if diff > 0:

        #negativo per allungamento percorso
        return - 1

    else:

        return 0


def compute_distance(vector,th = THRESHOLD):


    vector_over_th = vector[vector > th]

    vector_over_th = abs(vector_over_th - th)

    distance = sum(vector_over_th)

    return distance


def Reward_function(old_matrix_traffic, matrix_traffic_update, start_local_usage, new_local_usage,old_path,action_selected):


    val_1 = round(np.max(old_matrix_traffic) /np.max(matrix_traffic_update),3)

    #Calcolo val_2
    num = round(compute_distance(start_local_usage),4)
    den = round(compute_distance(new_local_usage),4)

    if num == 0 and den == 0:
        val_2 = 1
    elif num != 0 and den == 0:
        val_2 = 250
    else:
        val_2 = round(num/den,3)

    # print()
    # print("num: ", num, "den: ", den, "val_2: ",val_2)
    # print()

    #Global Reward

    if val_1 >= 1:
        #rew_1 = math.e**(2*(val_1-1))
        rew_1 = 0
    elif val_1 < 1:
        #rew_1 = -math.e**(2*(1/val_1-1))
        #rew_1 = -10
        rew_1 = -250
    else:
        rew_1 = 0

    #Local Reward
    # if val_2 > 1:
    #     rew_2 = math.e**(2*(val_2-1))
    # elif val_2 < 1:
    #     rew_2 = -math.e**(2*(1/val_2-1))
    # else:
    #     rew_2 = 0

    if val_2 == 250 or val_2 == 0:
        rew_2 = 250
    elif val_2 > 1:
        try:
            rew_2 = min(250,math.e**(val_2))
            #rew_2 = min(10,val_2)
            #rew_2 = val_2
        except:
            rew_2 = 100
    elif val_2 < 1:
        try :
            rew_2 = max(-250,-math.e**((1/val_2)))
            #rew_2 = max(-10,- 1/val_2)
            #rew_2 = - 1/val_2
        except:
            rew_2 = -250
    else:
        rew_2 = 0
        add = compute_newPath_length(old_path, action_selected)
        # print("Valutata variazione lunghezza path ",add)
        rew_2 += add

    #fixed value
    rew_3 = 0

    #reward_ = round(rew_1 + rew_2+val_3 ,4)
    if rew_1 == -250:
        #print("Globale NEGATIVA")
        reward_ = -250
    else:
        reward_ = round(rew_1 + rew_2 ,3)
        #print("Total Reward: ", reward_)
    # print("Reward: ", reward_)
    # print()
    # print()
    return rew_1, rew_2, rew_3, reward_


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


#inserisci controllo numero di oggetti salvati : numero di righe del df
def getCorrectFolder(save_folder, how_many_object):
    ret=-1
    ok_folders=[] # cartelle che contengono un potenziale backup
    last_timestamp=0
    last_folder=0

    for folder in ['1/', '2/']:
        checkfile=save_folder + folder + 'check'
        print('check file: ', checkfile)
        if Path.exists(Path(checkfile)): # controllo se esiste il file di check
            dfcheck=pd.read_csv(checkfile)
            print('ok df')

            #controllo che non ci siano nan e quindi il file sia stato scritto "tutto" e controllo che siano stati scritti tutti i file necessari
            if not dfcheck['timestamp'].isnull().values.any() and dfcheck.shape[0]==how_many_object:
                ok_folders.append(folder) #la cartella contine un backup )non sappiamo se aggiornato o no)

                # ultima rihga, ultimo timestamp
                if dfcheck.iloc[-1:]['timestamp'].item()>last_timestamp:
                    last_timestamp=dfcheck.iloc[-1:]['timestamp'].item() #last one time stamp
                    last_folder=folder


    if len(ok_folders)==0:
        ret=-1
    elif len(ok_folders)==1:
        ret=ok_folders[0]
    else:
        ret=last_folder

    return ret

def resume(marl, save_folder):

    how_many_object = 2 # number of oject in the check file
    folder=getCorrectFolder(save_folder, how_many_object) # obtaining the folder of the newes version
    print('correct_folder: ', folder)
    log_file = save_folder + 'log.txt'

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if folder==-1: # we have no backup
        print('No backup found \nGeneral Restarting ')
        with open(log_file, 'w') as target:
            print('No backup found \nGeneral Restarting ', file=target)
        return False, None, -1

    save_folder = save_folder+folder # updating the folder that contains the backup


    agents = [Path(save_folder+'agent_' + str(agent.id_node)) for agent in marl.agents] # file names of the agents
    i_game_score=Path(save_folder+'i_game_score') # file name of the i_game_score

    files = agents + [i_game_score] # whole files together

    check = all(list(map(Path.exists, files))) # cotrollo che tutti i file esistano

    if check:
      print('!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      with open(save_folder + 'log.txt', 'a+') as target:
          print('!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!!!!', file=target)

      agents_objs = []
      for agent_file in agents:
          with open(agent_file, 'rb') as f:
              agent = dill.load(f)
              agent.load_model(save_folder)
              agents_objs.append(agent)
      marl.agents=agents_objs

      with open(i_game_score, 'rb') as f:
        i_game_score=dill.load(f)

      return check, marl, i_game_score
    else:
        print('No backup found \nGeneral Restarting ')
        with open(save_folder + 'log.txt', 'a') as target:
            print('\n\nNo backup found \nGeneral Restarting ', file=target)
        return check, None, -1


def saveAll(marl, i_game_score, startingTime, save_folder):
    # print over the terminal
    print('saving')
    print('i game score ' + str(i_game_score), '/', lenIndexTrain)

    log_file = save_folder + 'log.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    #print over the file log.txt
    with open(save_folder + 'log.txt', 'a+') as target:
        print('saving', file=target)
        print('i game score ' + str(i_game_score), '/', lenIndexTrain, file=target)

    #marl.save_best_score() # updating the best scores pf the agents

    folders=['1/', '2/'] # controlled redundancy
    for folder in  folders:

        new_base=save_folder+folder # folder to save files
        checkfile=new_base +'check'  # in oder to have a file for checking the writing of all files


        #check if checkfile exist, create otherwise
        os.makedirs(os.path.dirname(checkfile), exist_ok=True)

        #inizialize the dataframe
        f=open(checkfile, 'w')
        f.write('file,timestamp\n')
        f.close()

        ###########################################################
        # how to save?
        # we use a csv file using a writing in 3 phase
        # phase 1: write just the name i.e. "we are tring to save this file"
        # phase 2: saving the file we are tring to save
        # phase 3: if we didn't crash jet we can write the timestam i.e.
        #            we was able to correctly write the file we are interested in
        #
        # we introduced a redundancy in order to always have a folder that contains a backup of our work
        # maybe it is same iteration back, but we are not loosing the whole work already done
        #
        # we neet time stamp since we are introducing redundacy and so we have the versioning problem
        # i.e. which copy is the newes?
        # the greatest the timestamp, the newest is the version
        ############################################################

        # files
        marl_f = new_base + 'marl'
        i_game_score_f=new_base+'i_game_score'

        #for marl
        f=open(checkfile, 'a')
        f.write(marl_f + ',') # writing the name of the objest ...
        f.close()

        marl.save(new_base) # ... saving the real file ...

        f=open(checkfile, 'a')
        f.write(str(time.time())+'\n') # .. writing the timestamp
        f.close()

        # for i_game_score

        f=open(checkfile, 'a')
        f.write(i_game_score_f+',')
        f.close()

        #open object file
        with open(i_game_score_f, 'wb') as f:
            #writing object file
            dill.dump(i_game_score, f)

        f=open(checkfile, 'a')
        f.write(str(time.time())+'\n')
        f.close()

    #print over the termina
    print('!!!! saved !!!!!')
    print("Episode: ", i_game_score)
    with open(save_folder + 'log.txt', 'a') as target:
        print('!!!! saved !!!!!', file=target)
        print("Episode: ", i_game_score, file=target)

    for agent in marl.agents:
        print("Agent: ", agent.id_node, "score: %.2f" % np.mean(agent.score),"best score: %.2f" % agent.best_score, "average_score %.2f" % agent.avg_score, "Epsilon: %.2f" %agent.epsilon)
        print("Last 20 scores: ", agent.scores[-20:])
        print('Time elapsed: ', str(dt.timedelta(seconds = time.time() - startingTime)))

        #print on log file
        with open(save_folder + 'log.txt', 'a') as target:
            print("Agent: ", agent.id_node, "score: %.2f" % np.mean(agent.score),"best score: %.2f" % agent.best_score, "average_score %.2f" % agent.avg_score, "Epsilon: %.2f" %agent.epsilon, file=target)
            print("Last 20 scores: ", agent.scores[-20:], file=target)
            print('Time elapsed: ', str(dt.timedelta(seconds = time.time() - startingTime)), file=target)


def create_agent_name(*argv, **kwargs):
    name = ''
    for key, value in list(kwargs.items()): #we start cocatenating _k_v for each couple of keyword args -> (k, v)
        name += "_%s_%s" %(key, value) # ciao

    for arg in argv: # we concatenate also the not keyword args
        name += '_' + arg

    return name[1:] #removed the fist (unusefull underscore)

def print_error(text):
    print('\033[91m' + 'Error: ' + text + '\033[0m')

def print_warning(text):
    print('\033[93m' + 'Warning: ' +  text + '\033[0m')

def check_agent_over_100(marl):
    ret=False
    for agent in marl.agents:
        if sum(agent.new_starting_local_out_prc_usage_orange > 1):
            print_error('Utilizazioni sopra il 100%')
            print(agent.id_node)
            print(agent.new_starting_local_out_prc_usage_orange > 1)
            print(agent.new_starting_local_out_prc_usage_orange)
            ret=True
    return ret

def get_agents_dict(agents_file):
    # return the dictionary in form {list_name : list} of the agents we want to implement.
    # the agents are writte in the agents_param.csv in the main folder
    agents_df = pd.read_csv(agents_file) # reading the dataframe

    dict = agents_df.to_dict(orient="list") # converting it to a dictionary
    print(dict)

    if agents_df.isnull().values.any(): # check if there is some columns with some nans
        print('The csv contain some Nans\nWe cannot proced')
        exit()

    #check if the df had all the necessary columns in order to build the agents
    necessary_cols = ['id_node_lst', 'alpha_lst', 'gamma_lst', 'epsilon_lst', 'epsilon_dec_lst', 'epsilon_end_lst','replace_target_lst', 'batch_size_lst', 'fcl1_lst', 'fcl2_lst']

    for col in necessary_cols:
        list = dict.get(col)
        if list == None:
            print('column ', col, 'dose not exist in the csv\nWe cannot proced')
            exit()

    return dict
