# MultiFDRnet
MultiFDRnet is a method for identifying significantly mutated subnetworks using multiplex PPI networks in human diseases.

## Setup

The setup process for MultiFDRnet includes the following steps:

### Download

Download MultiFDRnet. The following command clones the current MultiFDRnet repository from GitHub:

`git clone https://github.com/yangle293/MultiFDRnet.git`

### Installation

The following software is required for installing MultiFDRnet:

- Linux/Unix
- [Python (3.6)](www.python.org)
- [NumPy (1.17)](https://www.numpy.org)
- [NetworkX (2.2)](https://networkx.github.io/)
- CPLEX (12.8)

#### Installing CPLEX
Academic users can obtain a free, complete version of CPLEX via [IBM SkillsBuild Software Downloads](https://www.ibm.com/academic/home).

After the installation of CPLEX, you also need to install the CPLEX-Python modules. Run

    cd yourCPLEXhome/python/VERSION/PLATFORM

    python setup.py install

## Use

### Input
There are three sets of input files: a set of gene index files, a set of edge list files and a gene scores file.  For example, the following files define a network containing two connected nodes A and B, which have a score of 0.5 and 0.2, respectively.
#### Index-to-gene file
This file associates each node with an index:

    1  A
    2  B

##### Edge list file
This file defines a network by using the indices in the index-to-gene file:

    1    2

##### Gene-to-score file
This file associates each gene with a local FDR score:

    A 0.5
    B 0.2

If you have a list of p-values for individual genes, you can calculate local FDR scores using either the original R implementation (http://cran.r-project.org/web/packages/locfdr/), or the python-implementation of the locfdr method (https://github.com/leekgroup/locfdr-python). For ease of use, we provide a copy of the python implementation (`locfdr-python`) and a wrapper function `locfdr_compute.py` in `example` directory.

### Running
1. Compute local FDRs by running `example/locfdr_compute.py` script.

2. Run MultiFDRnet by running `src/MultiFDRnet.py` script.

See the `Examples` section for a full minimal working example of MultiFDRnet.

### Output
The output file is a plain text file organized as follows:

    Seed Gene   Running Time    Optimization Status Subnetwork
### Usage

  usage: multiplexFDRnet.py [-h] -igi INPUT_GENE_INDEX [INPUT_GENE_INDEX ...]
                          -iel INPUT_EDGE_LIST [INPUT_EDGE_LIST ...] -lw
                          LAYER_WEIGHT [LAYER_WEIGHT ...] -igl INPUT_GENE_LFDR
                          [-ofn OUTPUT_FILE_NAME] [-se SEED] [-fc FOCUS]
                          [-bd BOUND] [-sz SIZE] [-tl TIME_LIMIT]
                          [-rg RELATIVE_GAP]

### Program argument

    -h             Show help message and exit
    -igi           File names of the index-to-gene file
    -iel           File names of the input edge list file, must be the same order as in index-to-gene file
    -igl           File name of the gene-to-score file
    -ofn           File name of output
    -se            Seed gene names, either specify a seed gene name (e.g., TP53) OR set 'all_genes' to use all the genes with local FDRs less than FDR bound as seeds
    -lw            layer (network) weight for input networks, usually used when we are dealing with context-specific networks,
    -fc            specify the name of network we will focus on, usually the name of a context-specific network, must be the same order as in index-to-gene file
    -bd            FDR bound, default 0.1
    -sz            Local exploration size, default 400
    -tl            Time limit for each seed for solving MILP problem, default 100
    -rg            Relative gap in a MILP problem, default 0.01

### Examples
We provide a local FDRs files in the `example` directory: `TCGA_BLCA_MERGE.txt` generated by combining mutation-based p-values generated by MutSig2CV and copy number-based p-values generated by GISTIC2 using the TCGA breast cancer mutation data. Users can use their own p-values and run

    python example/locfdr_compute.py example/pvalues.txt

to calculate the local FDRs. The file of p-values has the same formulation with the local FDRs file.

There are two ways to use our algorithm. First, users can try to identify a subnetwork around any gene regardless of its local FDR score by using `-se gene_name` option (possibly no solution) to show a local picture of perturbation. Second, users can obtain a landscape of all perturbed regions in a PPI network by using `-se all_genes`. Running with this option will return a set of significantly perturbed subnetworks around all the seeds (i.e., genes with local FDR score less than the given bound B). See our paper for details. Here, we use the famous cancer gene PIK3CA as an example, that is, to identify a subnetwork around TP53 gene using four general-purpose PPI networks with equal layer weights, by running the following code:

    python src/MultiFDRnet.py -igi network_data/biogrid_index_gene network_data/irefindex18_index_gene network_data/reactome21_index_gene network_data/STRING_900_index_gene -iel network_data/biogrid_edge_list network_data/irefindex18_edge_list network_data/reactome21_edge_list network_data/STRING_900_edge_list -lw 1 1 1 1 -igl example/TCGA_BLCA_MERGE.txt -ofn subnetworks.txt -se PIK3CA

The output file look like this one:

| Seed Gene | Running Time  | Optimization Status | Subnetwork|
|:-------:|:-------:|:-----:|:------:|
| PIK3CA	|12.04720425605774	| optimal_tolerance	|FRS2 RAP1B PIK3CA FGFR3 F11R|

## Additional information
### Support
Please send us an email if you have any question when using MultiFDRnet.
### License
See `LICENSE.txt` for license information.
### Citation
If you use MultiFDRnet in your work, please cite the following manuscript:
L. Yang, R. Chen, Thomas Melendy, S. Goodison, Y. Sun. Identifying Significantly Disrupted Subnetworks Using Multiple Protein-Protein Interaction Networks in Cancer.

