import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as num


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
    print(avgdegree, variancedegree, mediana, sessantaperc, settantaperc)
    return avgdegree


#Pagerank
def pagerank(G, alpha, filename):
    r = nx.pagerank(G, alpha, None, 100, 1e-06, None, 'weight', None)
    f = open(filename, 'w')
    f.write(str(r))
    f.close()

def main():
    #Open Google dataset
    data_dataset = opendataset("/Users/mariaelena/Desktop/web-Google.txt")
    nodes = splitdatanodes(data_dataset)
    edges = splitdataedges(data_dataset)
    G = creategraph(nodes, edges)
    #print(G.is_directed())
    #numnodes = numbernodes(G)
    #numedges = numberedges(G)
    #numselfloops = numberselfloop(G)
    #print(numnodes, numedges, numselfloops)
    #degree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt', '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt', '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #max_in = maxdegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    #max_out = maxdegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    #max_tot = maxdegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #print(max_in, max_out, max_tot)
    #min_in = mindegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    #min_out = mindegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    #min_tot = mindegree(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #print(min_in, min_out, min_tot)
    #pagerank(G, 0.85, '/Users/mariaelena/Desktop/analisi_grafo/pagerank/pagerank.txt')
    #avg_degree_in = avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    #avg_degree_out = avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    #avg_degree_tot = avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    #print(avg_degree_in, avg_degree_out, avg_degree_tot)
    # avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_tot.txt')
    # avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_in.txt')
    # avgdegrees(G, '/Users/mariaelena/Desktop/analisi_grafo/grado/grado_out.txt')
    #print(nx.hits(G, max_iter=100, tol=1e-08, nstart=None, normalized=True))

if __name__ == "__main__":
    main()



