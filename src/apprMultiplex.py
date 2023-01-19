import networkx as nx
import numpy as np
from scipy.sparse import find,csr_matrix
import math
import heapq
from multiplex import Multiplex

def appr_multiplex_size(mx,seed,size,gamma=0.998):
    # input:
    # P: transistion csr_matrix
    # v: node volumn
    # S: seed set
    # size: size parameters
    # gamma: teleporation parameters
    P = mx.get_transition_matrix()
    v = mx.get_stationary() * mx.get_state_size()
    epsilon = 1
    e = np.array([0.0]*len(v)) # residue vector
    p = np.array([0.0]*len(v)) # appr vector

    gamma_bar = (1. - gamma)/ (1. + gamma) #convert to lazy-walk teleporation
    ## physical seeds
    if isinstance(seed,list):
        for s in seed:
            e[mx.get_state_node().index(s)] = 1./len(seed)
    else:
        e[mx.get_state_node().index(s)] = 1.
    Q = list(np.where(e/v > epsilon)[0])
    while len(np.where(p > 0.0)[0]) < size:
        if len(np.where(p > 0.0)[0]) != 0:
            step = size/float(len(np.where(p > 0.0)[0]))
        else:
            step = 2
        epsilon = epsilon/step
        Q = list(np.where(e/v > epsilon)[0])
        while(len(Q)>0):
            ind = Q.pop() # select a node to update
            e_bar = e[ind]
            p[ind] = p[ind] + gamma_bar * e_bar
            e[ind] = (1 - gamma_bar) * e_bar / 2.
            L = list(find(P[:,ind]>0)[0])
            for l in L:
                e[l] = e[l] + (1 - gamma_bar) * P[l,ind] * e_bar / 2.
            Q = list(np.where(e/v > epsilon)[0])
        p = p/v
    return dict(zip(mx.get_state_node(),p))



if __name__ == "__main__":
    g1 = nx.Graph()
    g1.add_edges_from([('a','b'),('b','c'),('d','e')])
    g2 = nx.Graph()
    g2.add_edges_from([('a','b'),('b','c'),('c','e')])
    graph_dict = {'alpha':g1,'beta':g2}
    fdr_dict = {'a':0.0,'b':0.0,'c':0.0}
    weight_dict = {'alpha':1.0,'beta':1.0}
    mx = Multiplex(graph_dict=graph_dict,fdr_dict=fdr_dict,weight_dict=weight_dict,interlayer_weight=1.0)

    Seed = ['a#alpha','a#beta']
    size = 10
    p = appr_multiplex_size(mx=mx,seed=Seed,size=size)
    nodes = heapq.nlargest(int(size), p, key=p.get)
    print(nodes)
