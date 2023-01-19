from __future__ import print_function
import cplex
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from cplex.exceptions import CplexError, CplexSolverError
import time
from scipy.sparse import find
import numpy as np
import networkx as nx
from multiplex import Multiplex

def solveMultiplexConductance(mx,local_genes,seed,fdr_bound,time_limit,relative_gap):
    fdr_data = mx.get_fdr()
    nodes = [x + '#' + y for x in local_genes for y in mx.get_layers()]
    local_index = [mx.get_state_node().index(x) for x in nodes]
    layers = mx.get_layers()
    adjacency_connection = mx.get_adjacency()[local_index,:][:,local_index]
    transition_multiplied = mx.get_trans_multiply_stat()[local_index,:][:,local_index]
    stationary_distribution = mx.get_stationary()[local_index]
    [conn_indi,conn_indj,conn_val] = find(adjacency_connection)
    [tran_indi,tran_indj,tran_val] = find(transition_multiplied)
    n_transition = len(tran_val);
    nNodes = len(nodes)
    nEdges = len(conn_val)
    u = 1.001 # upper bound of Z
    l = -0.001 # lower bound of Z
    tmp1 = [(x,y) for (x,y) in zip(tran_indi,tran_indj) if x<=y]
    tmp2 = [(y,x) for (x,y) in zip(tran_indi,tran_indj) if x>y]
    xx_entrys = list(set(tmp1 + tmp2))
    xx_entrys_value = [transition_multiplied[x,y] + transition_multiplied[y,x] for (x,y) in xx_entrys]

    ######## Variables Start
    # Variable Name
    variable_x = [x for x in nodes] # x_i, indicator variable
    variable_z = "obj" # Z
    variable_zx = ["".join(("z", x)) for x in nodes] # zx_i
    variable_xx = ["".join((np.array(nodes)[x],"*",np.array(nodes)[y])) for (x,y) in xx_entrys] # xx_ij #transition nonzero for conductance calculation
    variable_y = ["".join((x, "->", y)) for x, y in zip(np.array(nodes)[conn_indi],np.array(nodes)[conn_indj])] #y_ij #egdes for flow
    variable_source = "source" # source flow
    variable_s2x = ["".join((variable_source, "_", seed + "_" + layers[0]))] # source to seed
    my_colnames = variable_x + [variable_z] + variable_zx + variable_xx +  variable_y + [variable_source] + variable_s2x
    variable_dict = dict(zip(my_colnames,range(len(my_colnames)))) # variable map, speed up problem building of cplex
    # Variable Type: x_i binary, other continous
    my_ctype = "I" * nNodes + "C" * (nEdges + nNodes + len(xx_entrys) + 3)
    # Variable Bound: x {0,1}, obj 0-1, zx: 0-1, xx: 0-1, y: 0-nodes, source: 0-nodes, source2seed: 0-nodes
    my_ub = [1.0] * nNodes + [1.0] * (1 + nNodes + len(xx_entrys)) + [float(nNodes)] * (nEdges + 2)
    my_lb = [0.0] * nNodes + [0.0] * (1 + nNodes + len(xx_entrys) + nEdges + 2)
    ######### Variables End

    ######## Objective
    my_obj = [0.0] * nNodes + [1.0] + [0.0] * (nEdges + nNodes + len(xx_entrys) + 2) # objective: min Z
    ######## Objective End

    ######## Constraints Start
    my_row = []

    ### seed constrain  x_seed = 1
    tmp_con = [[variable_dict[seed + '#' + layers[0]]], [1.0]]
    my_row.append([[variable_dict[seed + '#' + layers[0]]], [1.0]])
    my_rhs_seed = [1.0]
    my_sense_seed = "E"
    my_rownames_seed = ["seed"]

    #### Budget constraint fdr: sum(lfdr-B) <= 0
    tmp_con = [ [variable_dict[x + '#' + layers[0]] for x in local_genes], [ float(fdr_data[x] - fdr_bound) for x in local_genes if x in fdr_data.keys()] ]
    my_row.append([ [variable_dict[x + '#' + layers[0]] for x in local_genes], [ float(fdr_data[x] - fdr_bound) for x in local_genes if x in fdr_data.keys()] ] )
    my_sense_budget_fdr = "L"
    my_rhs_budget_fdr = [0.0]
    my_rownames_budget_fdr = ["Budget_fdr"]

    #### Connectivity constraints
    # residual flow + flow injected into network = total flow  z0 + sum(y_zv) = N
    my_row.append([[variable_dict[variable_source], variable_dict[variable_s2x[0]]], [1.0] * (1 + 1)])
    my_sense_total = "E"
    my_rhs_total = [float(nNodes)]
    my_rownames_total = ["total"]

    # positive flow
    for i in range(nEdges):
        my_row.append([[variable_dict[variable_y[i]], variable_dict[variable_y[i].split("->")[-1]]], [1.0, -float(nNodes)]])
    my_rhs_positive = [0.0] * nEdges
    my_sense_positive = "L" * nEdges
    my_rownames_positive = ["".join((variable_y[i], "flow")) for i in range(nEdges)]

    # consuming flow
    my_rhs_consuming = []
    my_sense_consuming = "E" * nNodes
    my_rownames_consuming = []
    for i in range(nNodes):
        edge_incoming = [variable_dict["".join((nodes[y],"->",nodes[i]))] for y in list(adjacency_connection[i,:].nonzero()[1])]
        edge_ongoing = [variable_dict["".join((nodes[i],"->",nodes[y]))] for y in list(adjacency_connection[:,i].nonzero()[0])]
        if nodes[i] == (seed + '#' + layers[0]):
            edge_incoming.append(variable_dict[variable_s2x[0]])
        edge_ongoing.append(variable_dict[nodes[i]])
        my_row.append([edge_incoming + edge_ongoing, [1.0] * len(edge_incoming) + [-1.0] * (len(edge_ongoing))])
        my_rhs_consuming = my_rhs_consuming + [0.0]
        my_rownames_consuming = my_rownames_consuming + ["".join(("consuming_", nodes[i]))]

    # injected = consumed
    my_rhs_equal = [0.0]
    my_sense_equal = "E"
    my_rownames_equal = ["Equal"]
    tmp = [variable_dict[x] for x in variable_x]
    tmp.append(variable_dict[variable_s2x[0]])
    my_row.append([tmp, [1.0] * nNodes + [-1.0]])

    #### Helper constraints
    # Multiplex conductance: -sum(zx_i * D_ii) + sum(p*x_ii) - sum(P*p*x_ij) <= 0 and x_ii == x_i
    my_rhs_obj = [0.0]
    my_sense_obj = "L"
    my_rownames_obj = ["obj_main"]
    var_zx = [variable_dict[x] for x in variable_zx]
    var_x = [variable_dict[x] for x in variable_x]
    var_xx = [variable_dict[x] for x in variable_xx]
    tmpvar = []
    tmpvar.extend(var_zx + var_x + var_xx)

    value_zx = [-1.0*val for val in stationary_distribution.squeeze()] # stationary
    value_x = [1.0*val for val in stationary_distribution.squeeze()] # stationary
    value_xx = [-1.0*val for val in xx_entrys_value] # transition*stationary
    tmpvalue = []
    tmpvalue.extend(value_zx + value_x + value_xx)

    my_row.append([tmpvar,tmpvalue])

    # Constraints for zx:
    # 1. zx_i >= z - u(1-x_i) ===>   z +   x_i - zx_i <= u
    # 2. zx_i >= l*x_i        ===>       l*x_i - zx_i <= 0
    # 3. zx_i <= z - l(1-x_i) ===>  -z - l*x_i + zx_i <= -l
    # 4. zx_i <= u*x_i        ===>     - u*x_i + zx_i <= 0
    my_rhs_zx = [u] * nNodes + [0.0] * nNodes + [-1.0*l] * nNodes + [0.0] * nNodes
    my_sense_zx = "L" * 4 * nNodes
    my_rownames_zx = ["".join(("obj_helper_zx1_",nodes[i])) for i in range(nNodes)] + ["".join(("obj_helper_zx2_",nodes[i])) for i in range(nNodes)] + ["".join(("obj_helper_zx3_",nodes[i])) for i in range(nNodes)] + ["".join(("obj_helper_zx4_",nodes[i])) for i in range(nNodes)]

    constrain_zx_1 = [[[variable_dict[variable_z],variable_dict[variable_x[i]],variable_dict[variable_zx[i]]],[1.0,u,-1.0]] for i in range(nNodes)]
    constrain_zx_2 = [[[variable_dict[variable_x[i]],variable_dict[variable_zx[i]]],[l,-1.0]] for i in range(nNodes)]
    constrain_zx_3 = [[[variable_dict[variable_z],variable_dict[variable_x[i]],variable_dict[variable_zx[i]]],[-1.0,-l,1.0]] for i in range(nNodes)]
    constrain_zx_4 = [[[variable_dict[variable_x[i]],variable_dict[variable_zx[i]]],[-u,1.0]] for i in range(nNodes)]
    my_row.extend(constrain_zx_1 + constrain_zx_2 + constrain_zx_3 + constrain_zx_4)

    # Constraints for xx
    # 1. xx_ij <= x_i                ===>  xx_ij - x_i        <= 0
    # 2 xx_ij <= x_j               ===>  xx_ij - x_j        <= 0
    # 3 xx_ij >= x_i + x_j - 1     ===> -xx_ij + x_i + x_j  <= 1

    my_rhs_xx = [0.0] * (2 * len(variable_xx)) + [1.0] * len(variable_xx)
    my_sense_xx = "L" * (3 * len(variable_xx))
    my_rownames_xx = ["".join(("obj_helper_xx1_",x.split('*')[0],x.split('*')[1])) for x in variable_xx] + \
                    ["".join(("obj_helper_xx2_",x.split('*')[0],x.split('*')[1])) for x in variable_xx] + \
                    ["".join(("obj_helper_xx3_",x.split('*')[0],x.split('*')[1])) for x in variable_xx]

    constrain_1 = [[[variable_dict[x],variable_dict[x.split('*')[0]]],[1.0,-1.0]] for x in variable_xx]
    constrain_2 = [[[variable_dict[x],variable_dict[x.split('*')[1]]],[1.0,-1.0]] for x in variable_xx]
    constrain_3 = [[[variable_dict[x],variable_dict[x.split('*')[0]],variable_dict[x.split('*')[1]]],[-1.0,1.0,1.0]] for x in variable_xx]
    my_row.extend(constrain_1 + constrain_2 + constrain_3)

    # Adherent constraint
    # gene_test1 = gene_test2 = gene_test3
    my_rhs_adh = [0.0] * len(local_genes) * (len(layers)-1)
    my_sense_adh = "E" * len(local_genes) * (len(layers)-1)
    my_rownames_adh = ["".join(("adh_",x,"")) for x in local_genes for i in range(1,len(layers))]
    constrain_adh = [[[variable_dict[x + '#'+layers[0]],variable_dict[x + '#' + layers[i]]],[1.0,-1.0]] for x in local_genes for i in range(1,len(layers))]
    my_row.extend(constrain_adh)

    #### Merge Constraints
    my_sense = my_sense_seed + my_sense_budget_fdr + my_sense_total + my_sense_positive + my_sense_consuming + my_sense_equal + \
                my_sense_obj + my_sense_zx + my_sense_xx + my_sense_adh
    my_rhs = my_rhs_seed + my_rhs_budget_fdr + my_rhs_total + my_rhs_positive + my_rhs_consuming + my_rhs_equal + \
                my_rhs_obj + my_rhs_zx + my_rhs_xx + my_rhs_adh
    my_rownames = my_rownames_seed + my_rownames_budget_fdr + my_rownames_total + \
                    my_rownames_positive + my_rownames_consuming + my_rownames_equal + my_rownames_obj + my_rownames_zx + my_rownames_xx + my_rownames_adh
    ######## Constraints End
    #print(len(my_row),len(my_sense),len(my_rhs),len(my_rownames))
    ######## Problem Build Start
    try:
        build_start = time.time()
        subnetwork = cplex.Cplex()
        subnetwork.parameters.timelimit.set(time_limit)
        subnetwork.parameters.mip.tolerances.mipgap.set(relative_gap)
        subnetwork.set_log_stream(None)
        subnetwork.set_warning_stream(None)
        subnetwork.set_results_stream(None)
        #subnetwork.parameters.simplex.tolerances.markowitz.set(0.99999)
        #subnetwork.parameters.read.scale.set(1)
        subnetwork.objective.set_sense(subnetwork.objective.sense.minimize)
        subnetwork.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype, names=my_colnames)
        subnetwork.linear_constraints.add(lin_expr=my_row, senses=my_sense,rhs=my_rhs, names=my_rownames)
        #build_end = time.time()
        #print("build time: ", build_end - build_start)
        #solve_start = time.time()
        subnetwork.solve()
        #solve_end = time.time()
        #print("Solve time: ", solve_end - solve_start)
    except CplexError as exc:
        print(exc)
        subnetwork_result = [];
        status = -1;
        return
    ######## Problem Build End

    ######## Solution Start
    solution = subnetwork.solution
    #solution.get_status() returns  an integer code
    #print("Solution status = ", solution.get_status(), ":", end=' ')
# # # the following line prints the corresponding string
# #   print(solution.status[solution.get_status()])
    status = solution.status[solution.get_status()]
    try:
        #print("Objective value = ", solution.get_objective_value())
        x = solution.get_values(0, subnetwork.variables.get_num() - 1)
        result = dict(zip(my_colnames, x))
    except CplexSolverError as exc:
#     #print(exc)
        subnetwork_result = []
        return subnetwork_result,status
# #print(result)
    subnetwork_result = [x for x in variable_x if result[x] > 0.9]
    return subnetwork_result, status


if __name__ == '__main__':
    database_list = ['test1','test2','test3']
    data_path = 'toy_data/'
    index_file_list = ['test1_index_gene','test2_index_gene','test3_index_gene']
    index_file_list_full = [data_path + x for x in index_file_list]
    edge_file_list = ['test1_edge_list','test2_edge_list','test3_edge_list']
    edge_file_list_full = [data_path + x for x in edge_file_list]
    score_file = 'test_fdr.txt'
    score_file_full = data_path + score_file
    seed = 'TCGA'
    fdr_bound = 0.1
