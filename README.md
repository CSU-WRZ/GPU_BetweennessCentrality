## Introduction

This is the source code for the algorithm proposed in our paper.

## Algorithm  
1. **QBC** (Query-based Betweenness Centrality)
2. **MHL-QBC** (MHL-Index QBC)
3. **RQ** (Reduced Query): query over a smaller set of vertices



## Data

All data is stored in the **data** folder.

- **data/roadData** folder: Contains two datasets,  LosAngeles and bay. 
  - The LosAngeles dataset is relatively smaller in scale, whereas the bay dataset is larger.
  - Format: The first line stores the number of nodes and edges. The following lines store Node ID1, Node ID2, and the edge length. 
  - Other road network data can be obtained from DIMACS [1] and OpenStreetMap [2].


- **data/orderFile** folder: Stores all node contraction orders. 
  - Format: The first line indicates the total number of nodes, and the following lines store the Node ID and its corresponding contraction order. (Order file is only used in the QBC algorithm to construct the shortest distance index ( H2H [3], P2H [4]) .)

- **data/queryBatch** folder: Contains the set of selected vertices for computing BC.
  - Format: The first line indicates the number of vertices to query, and the following lines store the Node ID.

[1] "9th DIMACS Implementation Challenge," http://www.diag.uniroma1.it/~challenge9/download.shtml.
[2] "OpenStreetMap," https://www.openstreetmap.org.
[3] Ouyang D, Qin L, Chang L, et al. When hierarchy meets 2-hop-labeling: Efficient shortest distance queries on road networks[C]//Proceedings of the 2018 International Conference on Management of Data. 2018: 709-724.
[4] Chen Z, Fu A W C, Jiang M, et al. P2H: Efficient distance querying on road networks by projected vertex separators[C]//Proceedings of the 2021 International Conference on Management of Data. 2021: 313-325.


## Compile


```c++

// Our server is running NVIDIA driver version 550.107.02, and CUDA version 12.4 (NVIDIA RTX 4090 24GB GPU). 

nvcc -std=c++17 -O3 -arch=sm_86  -ccbin /usr/bin/g++-10   -Xcompiler -fopenmp QBC.cu  src/Graph.cpp  src/buildIndex.cpp  -o QBC


nvcc -std=c++17 -O3 -arch=sm_86  -ccbin /usr/bin/g++-10  MHLQBC.cu  src/TreeCount/TreeDecomp.cpp src/TreeCount/Graph.cpp src/Graph.cpp src/buildIndex.cpp  -o MHLQBC


nvcc -std=c++17 -O3 -arch=sm_86  -ccbin /usr/bin/g++-10  RQ.cu  src/TreeCount/TreeDecomp.cpp src/TreeCount/Graph.cpp src/Graph.cpp  src/buildIndex.cpp  -o RQ


```


## Run

#### Run **QBC** Algorithm
- We run the QBC code by inputing the path of the **graph file**, node contraction **order file**, **device ID** used for GPU executions, and **output file path** where the computed BC values will be saved.
- For example, 

```shell

# to run the algorithm on the **LosAngeles** road network with **device ID = 0**, and save the results to **LosAngelesBC.txt**, you can use the following command:
./QBC data/roadData/LosAngeles.txt data/orderFile/LosAngelesOrder.txt 0 LosAngelesBC.txt


# to run the algorithm on the **bay** road network with **device ID = 0**, and save the results to **bayBC.txt**, you can use the following command:
./QBC data/roadData/bay.txt data/orderFile/bayOrder.txt 0 bayBC.txt

```

#### Run **MHL-QBC** Algorithm
- We run the MHL-QBC code by inputing the path of the **graph file**, **device ID** used for GPU executions, and **output file path** where the computed BC values will be saved.
- For example, 
```shell

# to run the algorithm on the **LosAngeles** road network with **device ID = 0**, and save the results to **LosAngelesBC.txt**, you can use the following command:
./MHLQBC data/roadData/LosAngeles.txt 0 LosAngelesBC.txt


#  to run the algorithm on the **bay** road network with **device ID = 0**, and save the results to **bayBC.txt**, you can use the following command:
./MHLQBC data/roadData/bay.txt 0 bayBC.txt

```


#### Run **RQ** Algorithm
- We run the RQ code by inputing the path of the **graph file**, **queried vertex file**, and **device ID** used for GPU executions.
- For example, 

```shell

# to run the algorithm on the **LosAngeles** road network with **device ID = 0**, you can use the following command:
./RQ data/roadData/LosAngeles.txt data/queryBatch/LosAngeles.txt 0


# to run the algorithm on the **bay** road network with **device ID = 0**, you can use the following command:
./RQ data/roadData/bay.txt data/queryBatch/bay.txt 0

```

## Index Construction

The GPU implementation extends from G2H, available at [https://github.com/CSU-WRZ/MHL-GPU](https://github.com/CSU-WRZ/MHL-GPU).
