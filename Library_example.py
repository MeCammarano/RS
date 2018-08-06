import sys
import json
from math import log
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as num
import powerlaw as powerlaw
import pylab as p
import scipy


#Load dataset
def opendataset( name ):
#"/Users/mariaelena/Desktop/web-Google.txt"
    file_graph = open(name, "r")
    data_dataset= file_graph.readlines()
    file_graph.close()
    return data_dataset

#Split data and return nodes
def splitdatanodes( data ):
    da = [i.split('\t', 1)[0] for i in data]
    return da

#Split data and return edges
def splitdataedges( data ):
    da = [i.split('\t', 1)[0] for i in data]
    a = [i.split('\t', 1)[1] for i in data]
    a2 = [i.split('\n')[0] for i in a]

    l = {(da[0], a2[0])}
    i = 1
    while i < len(data):
        l.add((da[i], a2[i]))
        i = i + 1
    return l

#Create Graph
def creategraph ( nodes, edges ):
   g1 = nx.DiGraph()
   g1.add_nodes_from(nodes)
   g1.add_edges_from(edges)
   return g1


#Return the number of nodes
def numbernodes(G):
    num = nx.number_of_nodes(G)
    return num

#Return the nummber of edges
def numberedges(G):
    num = nx.number_of_edges(G)
    return num

#Return the number of selfloops
def numberselfloop(G):
    num = nx.number_of_selfloops(G)
    return num

#Compute and write in the files the degree
def degree(G, filename1, filename2, filename3):
    #Out degree for all nodes
    f = open(filename1, 'w')
    f.writelines(["(%s, %d)\n" % item for item in G.out_degree()])
    f.close()
    #In degree for all nodes
    f1 = open(filename2, 'w')
    f1.writelines(["(%s, %d)\n" % item for item in G.in_degree()])
    f1.close()
    #Total degree for all nodes
    f2 = open(filename3, 'w')
    f2.writelines(["(%s, %d)\n" % item for item in G.degree()])
    f2.close()

#Compute the max degree of a Graph
def maxdegree(G, filename):
    f = open(filename, 'r')
    line = f.readlines()
    cont = [i.split(', ', 1)[1] for i in line]
    cont2 = [j.split(')\n')[0] for j in cont]
    f.close()
    massimo = max(cont2, key=lambda x: int(x))
    return massimo

#Compute the min degree of a Graph
def mindegree(G, filename):
    f = open(filename, 'r')
    line = f.readlines()
    cont = [i.split(', ', 1)[1] for i in line]
    cont2 = [j.split(')\n')[0] for j in cont]
    f.close()
    minimo = min(cont2, key=lambda x: int(x))
    return minimo

#Compute the avg degree of a Graph
def avgdegrees(G, filename):
    #sum(degrees)/len(degrees)
    f = open(filename, 'r')
    line = f.readlines()
    cont = [i.split(', ', 1)[1] for i in line]
    cont2 = [j.split(')\n')[0] for j in cont]
    numbers = [int(x) for x in cont2]
    avgdegree = sum(numbers)/len(numbers)
    variancedegree = sum(map(lambda x: (x - avgdegree)**2, numbers)) / len(numbers)
    mediana = num.ceil(num.percentile(numbers, 50))
    sessantaperc = num.ceil(num.percentile(numbers, 60))
    settantaperc = num.ceil(num.percentile(numbers, 70))
    #print(avgdegree)
    print(variancedegree)
    print(mediana, sessantaperc, settantaperc)
    return avgdegree


#Pagerank
def pagerank(G, alpha, filename):
    r = nx.pagerank(G, alpha, None, 100, 1e-06, None, 'weight', None)
    f = open(filename, 'w')
    f.write(str(r))
    f.close()
    minimo = min(r.items(), key = lambda x: x[0])
    massimo = max(r.items(), key = lambda x: x[0])
    numbers = [x for x in r.values()]
    media = sum(numbers)/len(numbers)
    print(minimo, massimo, media)

#HITS
def hits(g):
    k, v =  nx.hits_numpy(g, True)
    minimo = min(k.items(), key=lambda x: x[1])
    massimo = max(k.items(), key=lambda x: x[1])
    media = sum(k.values())/len(k)
    print(minimo, massimo, media)
    minimo_a = min(v.items(), key=lambda x: x[1])
    massimo_a = max(v.items(), key=lambda x: x[1])
    media_a = sum(v.values())/len(v)
    print(minimo_a, massimo_a, media_a)

#Power Law
def power_law_method(g):
    #si prendono solo i gradi di ciascun nodo
    degree = [t[1] for t in g.in_degree]
    #trasforma la lista con series
    degree_cont = pd.Series(degree).value_counts()
    #calcola la frequenza di ogni termine
    d1 = {d: degree_cont[d] / sum(degree_cont) for d in degree_cont.index}
    #si prendono le chiavi
    x = d1.keys()
    #si prendono i valori
    y = d1.values()
    fit = powerlaw.Fit(degree, xmin=min(degree), xmax=max(degree))
    plt.rc('text', usetex=True)
    fig = plt.figure()
    print(fit.power_law.alpha)
    print(fit.power_law.sigma)
    print(fit.distribution_compare('power_law', 'exponential'))
    #powerlaw.plot_pdf(degree_count, linear_bins=True, color='r')
    list_x = list(x)
    #print(list_x)
    list_y = list(y)
    #print(list_y)
    #si effettua il logaritmo sia delle chiavi che dei valori per poterli plottare
    list_log_x = num.log10(list_x)
    list_log_y = num.log10(list_y)
    #si plottano i risultati
    plt.plot(list_log_x, list_log_y, 'r+')
    #si calcola l'alpha
    alpha = fit.power_law.alpha
    power_x_value = num.linspace(min(degree_cont.index), max(degree_cont.index), 100)
    #x elevato alla alpha
    power_y_value = list(map(lambda x: x**(-alpha), power_x_value))
    power_x_value = num.log10(power_x_value)
    power_y_value = num.log10(power_y_value)
    plt.plot(power_x_value, power_y_value, 'b-')
    plt.savefig("/Users/mariaelena/Desktop/img/prima.png".format('ciao'), dpi=fig.dpi)
    plt.show()



#TSS
def tss(g, soglia):
        target_set = set()
        #dizionario le cui chiavi sono i nodi e i valori le soglie relative a ciascun nodo
        g1 = g
        soglie = dict()
        insieme_soglia = {node: soglia for node in g1.nodes}

        while len(g1.nodes) != 0:
          #Primo caso
          #insieme dei nodi la cui soglia è pari a 0
          soglia_vuota = list(filter(lambda n1: insieme_soglia == 0, g1.nodes))
          if(len(soglia_vuota) > 0):
              #scorro i nodi interessati
              for nodo in soglia_vuota:
                  #scorro i vicini dei nodi interessati
                  for vicino in g1.neighbors(nodo):
                      #diminuisco di 1 la soglia dei vicini (vedere paper, versione greedy)
                      insieme_soglia[vicino] = max(insieme_soglia[vicino] - 1, 0)
              g1.remove_nodes_from(soglia_vuota)
          else:
              #controllo se la soglia del nodo è maggiore rispetto al suo grado (vedere paper, versione greedy)
              high_soglia = max(g1.nodes, key=lambda n: insieme_soglia[n]-g1.degree[n])
              if insieme_soglia[high_soglia] > g1.degree[high_soglia]:
                  #aggiungo alla soluzione
                  target_set.add(high_soglia)
                  #scorro i vicini
                  for vicino in g1.neighbors(high_soglia):
                      #diminuisco di 1
                      insieme_soglia[vicino] = insieme_soglia[vicino] - 1
                  #rimuovo dal grafo
                  g1.remove_node(high_soglia)
              else:
                  v = max(g1.nodes, key=lambda n: insieme_soglia[n]/(g1.degree[n] * (g1.degree[n] + 1)))
                  g1.remove_node(v)

        return target_set


def main():
    #Open Google dataset
    data_dataset = opendataset("/Users/mariaelena/Desktop/p2p-Gnutella05.txt")
    nodes = splitdatanodes(data_dataset)
    edges = splitdataedges(data_dataset)
    G = creategraph(nodes, edges)
    #print(G.is_directed())
    #numnodes = numbernodes(G)
    #numedges = numberedges(G)
    #numselfloops = numberselfloop(G)
    #print(numnodes, numedges, numselfloops)
    #degree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt', '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt', '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    # max_in = maxdegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    # max_out = maxdegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    # max_tot = maxdegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #print(max_in, max_out, max_tot)
    # min_in = mindegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    # min_out = mindegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    # min_tot = mindegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #print(min_in, min_out, min_tot)
    #pagerank(G, 0.85, '/Users/mariaelena/Desktop/analisi_grafo/pagerank/pagerank.txt')
    # avg_degree_in = avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    # avg_degree_out = avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    # avg_degree_tot = avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #print(avg_degree_in, avg_degree_out, avg_degree_tot)
    #avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    #avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    #nx.hits(G, 100, 1e-08, None, True))
    #hits(G)
    #print(tss(G, 2))
    #power_law_method(G)


if __name__ == "__main__":
    main()




