from __future__ import print_function
from multiplex import Multiplex
from apprMultiplex import appr_multiplex_size
import networkx as nx
import time
import heapq
import argparse
import sys
from solveMultiplexConductance import solveMultiplexConductance
from dataLoad import load_network_data, load_fdr_scores
import os
import re
import operator

# Parse arguments.
def get_parser():
    description = 'MultiFDRnet: identifying significant mutated subnetworks using multiple PPI networks.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-igi','--input_gene_index',type=str,nargs='+',required=True,help='Input gene index file names')
    parser.add_argument('-iel','--input_edge_list',type=str,nargs='+',required=True,help='Input edge list file names')
    parser.add_argument('-lw','--layer_weight',type=int,nargs='+',required=True,help='Input layer weights')
    parser.add_argument('-igl','--input_gene_lfdr',type=str,required=True,help='Input gene local FDRs')
    parser.add_argument('-ofn','--output_file_name',type=str,required=False,default='subnetworks.txt',help='File name of output')
    parser.add_argument('-se','--seed',type=str,required=False,default='all_genes',help='Seed gene name')
    parser.add_argument('-fc','--focus',type=str,required=False,default='None',help='Focused network name')
    parser.add_argument('-bd','--bound',type=float,required=False,default='0.1',help='FDR bound')
    parser.add_argument('-sz','--size',type=int,required=False,default=400,help='Local exploration size')
    parser.add_argument('-tl','--time_limit',type=int,required=False,default=100,help='Time limit for each seed')
    parser.add_argument('-rg','--relative_gap',type=float,required=False,default=0.01,help='Relative gap in MILP')
    return parser




def multiplexFDRnetMain(mx,graph_dict,seed,focus,fdr_bound,size=400,time_limit=100,relative_gap=0.01):
    print('start for',seed)
    ## get seed list
    if seed == 'all_genes':
        seed_list = [x for x in mx.get_fdr().keys() if mx.get_fdr()[x] <= fdr_bound]
        if focus != 'None':
            seed_list = [x for x in seed_list if x in list(graph_dict[focus])]
        agg_graph = mx.get_agg_graph()
        seed_subnetwork = agg_graph.subgraph(seed_list) # seeds subnetwork
        seed_data = [(x,seed_subnetwork.degree(x)) for x in seed_list]
        seed_sort_neighbor = sorted(seed_data, key=operator.itemgetter(1),reverse=True)
        print(seed_sort_neighbor)
        seed_list = [x for (x,y) in seed_sort_neighbor]
    else:
        seed_list = [seed]
    result = []
    count = 0
    selected_seed = set() # maintain a set of selected genes
    for s in seed_list:
        count = count + 1
        print("Searching subnetwork for ",s," ",str(count),"/",str(len(seed_list)))
        if s in selected_seed:
            print("This gene is already included...")
            continue
        print("Extracting local graph for ",s)
        # local graph with state_node
        seed_state = [s + '#' + l for l in mx.get_layers()]
        normalized_ppr = appr_multiplex_size(mx=mx,seed=seed_state,size=size)
        nodes = heapq.nlargest(int(size), normalized_ppr, key=normalized_ppr.get)
        # add all physical related to state_node
        local_genes = list(set([x.split('#')[0] for x in nodes]))
        print("Size of local graph:",len(local_genes))
        ## solve conductance minimization problem
        seed_start = time.time()
        print("Solving Optimization Problem for ",s)
        result_state_nodes,status = solveMultiplexConductance(mx=mx, local_genes = local_genes, seed=s, fdr_bound=fdr_bound,time_limit=time_limit,relative_gap=relative_gap)
        seed_end = time.time()
        result_genes = list(set([x.split('#')[0] for x in result_state_nodes]))
        selected_seed = selected_seed.union(result_genes)
        result.append((s, seed_end-seed_start, status, result_state_nodes,result_genes))
    return result


def run(args):
    print('Loading network data...')
    # network_path = 'network_data/'
    # index_file_list = [network_path + network + '_index_gene' for network in networkList]
    # edge_file_list = [network_path + network + '_edge_list' for network in networkList]
    networkList = [n.split('_index_gene')[0].split('/')[-1] for n in args.input_gene_index]
    graph_dict = load_network_data(networkList,args.input_gene_index,args.input_edge_list)
    # fdr
    #fdr_data = 'fdrdata/' + cancer
    print('Loading local FDR data...')
    fdr_dict = load_fdr_scores(args.input_gene_lfdr)
    #weight = [1.0] * len(networkList)
    weight_dict = dict(zip(networkList,args.layer_weight))
    # build multiplex
    print('Building multiplex...')
    mx = Multiplex(graph_dict,fdr_dict,weight_dict)

    print('Running optimization...')
    result = multiplexFDRnetMain(mx,graph_dict=graph_dict,seed=args.seed,focus=args.focus,fdr_bound=args.bound,size=args.size,time_limit=args.time_limit,relative_gap=args.relative_gap)

    ## save result
    #result_file = "".join(("result/","".join(networkList),cancer, "_bound", str(bound),"_s",str(size),"tl",str(time_limit),"rg",str(relative_gap),"_multiplexFDRnet.txt"))
    result_file = args.output_file_name
    this_folder = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(this_folder, result_file)
    print('Saving results...')
    with open(my_file, 'w') as outfile:
        outfile.write("".join(("Seed Gene", "\t", "Running Time","\t","Optimization Status","\t","Subnetwork", "\n")))
        for (seed,time,status, r_s,r) in result:
            outfile.write("".join((seed, "\t", str(time),"\t",status,"\t"," ".join(r), "\n")))









if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
