from __future__ import print_function
import networkx as nx

## load single networks and fdr scores
## save in a list of graphs

def load_network_data(database_list,index_file_list,edge_file_list):
    assert len(database_list) == len(edge_file_list)
    assert len(index_file_list) == len(edge_file_list)
    graph_dict = {}

    for (database,ind,edge) in zip(database_list,index_file_list,edge_file_list):
        g = load_single_network_from_file(ind, edge)
        graph_dict[database] = g
    return graph_dict

def load_single_network_from_file(index_file, edge_file):
    # Load gene-index map
    with open(index_file) as infile:
        arrs = [l.rstrip().split() for l in infile]
        indexToGene = dict((int(arr[0]), arr[1]) for arr in arrs)
    G = nx.OrderedGraph()
    G.add_nodes_from(indexToGene.values())  # in case any nodes have degree zero
    # Load graph
    with open(edge_file) as infile:
        edges = [map(int, l.rstrip().split()[:2]) for l in infile]
    G.add_edges_from([(indexToGene[u], indexToGene[v]) for u, v in edges])
    selfLoops = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(selfLoops)
    return G

def load_fdr_scores(score_file):
    with open(score_file) as infile:
        arrs = [l.rstrip().split() for l in infile]
        geneToScores = dict((arr[0], float(arr[1])) for arr in arrs)
    #fdr_dict = {}
    #for g in genes:
#        if g in geneToScores.keys():#
#            fdr_dict[g] = geneToScores[g]
#        else:
#            fdr_dict[g] = 0.0
    return geneToScores


if __name__ == "__main__":
    ## test data loading
    database_list = ['test1','test2','test3']
    data_path = 'toy_data/'
    index_file_list = ['test1_index_gene','test2_index_gene','test3_index_gene']
    index_file_list_full = [data_path + x for x in index_file_list]
    edge_file_list = ['test1_edge_list','test2_edge_list','test3_edge_list']
    edge_file_list_full = [data_path + x for x in edge_file_list]
    score_file = 'test_fdr.txt'
    score_file_full = data_path + score_file
    G_dict = load_network_data(database_list,index_file_list_full,edge_file_list_full)
    fdr_dict = load_fdr_scores(score_file_full)

    ## print
    for g in G_dict.keys():
        print('Graph: ',g)
        print(G_dict[g].nodes)
        print(G_dict[g].edges)
    print('Local FDR scores:',fdr_dict)
