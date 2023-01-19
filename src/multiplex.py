##build a multiplex and calculate transition matrix from a list of graphs
import networkx as nx
import numpy as np
import itertools
from scipy.sparse import csr_matrix, find, identity, lil_matrix
from scipy.sparse.linalg import eigs,bicgstab
import numpy.matlib as npmat


def largest_component_multiplex(graph_dict):
    agg_graph = nx.Graph()
    for key in graph_dict.keys():
        agg_graph.add_edges_from(graph_dict[key].edges)
    lcc = max(nx.connected_components(agg_graph),key=len)
    graph_dict_largest_component = {}
    for key in graph_dict.keys():
        graph_dict_largest_component[key] = graph_dict[key].subgraph(lcc).copy()
    return lcc, graph_dict_largest_component,agg_graph

def page_rank(P,alpha,s):
    print('Solving eigs...')
    s = s/np.sum(s);
    if alpha == 0.0:
        [v,p] = eigs(P,k=1,sigma=1,which='LM')
        p = np.real(p)
    else:
        #scipy.sparse.linalg.bicgstab(A, b, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)
        p,info=bicgstab((identity(P.shape[0],format='csr')+(alpha-1)*P),alpha*s,maxiter=100);
    p = p/np.sum(p)
    return p


class Multiplex:

    def __init__(self, graph_dict, fdr_dict, weight_dict,interlayer_weight=1.0):
        self.layers = list(graph_dict.keys()) # all the layers name
        assert graph_dict.keys() == weight_dict.keys()
        self.physical, tmp_graph_dict, self.agg_graph =  largest_component_multiplex(graph_dict)# all the physical nodes from LLC based on segregated graph
        self.state_node = [x + '#' + key for key in self.layers for x in self.physical ]
        self.supra_graph = nx.OrderedGraph()
        self.supra_graph.add_nodes_from(self.state_node)
        ## add intra-layer edges
        for key in tmp_graph_dict.keys():
            self.supra_graph.add_edges_from([(x[0] + '#' + key, x[1] + '#' + key,{'weight':weight_dict[key]}) for x in tmp_graph_dict[key].edges])
        ## add inter-layer edges
        layer_pairs = itertools.combinations(self.layers,2)
        for layer1,layer2 in layer_pairs:
            self.supra_graph.add_edges_from([(n + '#' + layer1, n + '#' + layer2, {'weight': interlayer_weight}) for n in self.physical])
        self.fdr_data = {}
        for p in self.physical:
            if p in fdr_dict.keys():
                self.fdr_data[p] = fdr_dict[p]
            else:
                self.fdr_data[p] = 1.0 #local FDR = 1 if no input
        # free memory
        del tmp_graph_dict
        self.adj_noweight = nx.adjacency_matrix(self.supra_graph)
        ## calculate transition csr_matrix
        adj = nx.adjacency_matrix(self.supra_graph,weight='weight')
        kout = np.asarray(np.sum(adj,axis=0)).flatten()
        [row,col,val]=find(adj)
        self.transition = csr_matrix((val/kout[col], (row, col)), shape=adj.shape)
        kin = np.asarray(np.sum(adj,axis=1)).flatten()
        self.stationary = page_rank(P = self.transition,alpha=0.05,s=kin)
        self.tran_multiply_stat = (self.transition.multiply(self.stationary.transpose())).transpose().tocsr()

    def get_transition_matrix(self):
        return self.transition

    def get_stationary(self):
        return self.stationary

    def get_supra_graph(self):
        return self.supra_graph

    def get_layers(self):
        return self.layers

    def get_state_node(self):
        return self.state_node

    def get_genes(self):
        return self.physical

    def get_fdr(self):
        return self.fdr_data

    def get_adjacency(self):
        return self.adj_noweight

    def get_state_size(self):
        return len(self.state_node)

    def get_trans_multiply_stat(self):
        return self.tran_multiply_stat

    def get_agg_graph(self):
        return self.agg_graph




if __name__ == "__main__":
    g1 = nx.Graph()
    g1.add_edges_from([('a','b'),('b','c'),('d','e')])
    g2 = nx.Graph()
    g2.add_edges_from([('a','b'),('b','c'),('c','e')])
    graph_dict = {'alpha':g1,'beta':g2}
    fdr_dict = {'a':0.0,'b':0.0,'c':0.0}
    weight_dict = {'alpha':1.0,'beta':1.0}
    mx = Multiplex(graph_dict=graph_dict,fdr_dict=fdr_dict,weight_dict=weight_dict,interlayer_weight=1.0)
    print('layer:',mx.layers)
    print('physical:',mx.physical)
    print('State node:',mx.state_node)
    print('Graph nodes:',mx.supra_graph.nodes)
    print('Graph edges:',mx.supra_graph.edges)
    print('fdr:',mx.fdr_data)

    print('Classic walk with omega=1 and alpha=0:')
    t,s,m = mx.get_transition_matrix()
    [row,col,val] = find(t)
    print('transition matrix:')
    for (x,y,z) in zip(row,col,val):
        print(mx.state_node[x],mx.state_node[y],z)
    print('stationary distribution:')
    for (x,y) in zip(mx.state_node,s):
        print(x,y)
    print('transition matrix multiplied:')
    [row1,col1,val1] = find(m)
    for (x,y,z) in zip(row1,col1,val1):
        print(mx.state_node[x],mx.state_node[y],z)
