
#include "include/Graph.h"
#include "omp.h"
#include <cstdio>
#include <cstdlib>

const int threadsPerBlock = 256, blocksPerGrid = 256, blockNum = 256;
const int numOfThread = 32;
string roadDataPath, orderFile;

void Graph::moveDisIndexToDevice() {

  int tmpSize = 0;
  // device_toRMQ
  tmpSize = toRMQ.size() * sizeof(int);
  cudaMalloc(&d_toRMQ, tmpSize);
  cudaMemcpy(d_toRMQ, toRMQ.data(), tmpSize, cudaMemcpyHostToDevice);

  // device_RMQIndex
  vector<int> h_RMQIndex_flat;
  vector<int> h_RMQIndex_flat_idx;
  h_RMQIndex_flat_idx.push_back(0);
  for (int i = 0; i < RMQIndex.size(); ++i) {
    for (int j = 0; j < RMQIndex[i].size(); ++j) {
      h_RMQIndex_flat.push_back(RMQIndex[i][j]);
    }
    h_RMQIndex_flat_idx.push_back(h_RMQIndex_flat.size());
  }
  tmpSize = h_RMQIndex_flat.size() * sizeof(int);
  cudaMalloc(&d_RMQIndex, tmpSize);
  cudaMemcpy(d_RMQIndex, h_RMQIndex_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_RMQIndex_flat_idx.size() * sizeof(int);
  cudaMalloc(&d_RMQIndex_idx, tmpSize);
  cudaMemcpy(d_RMQIndex_idx, h_RMQIndex_flat_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  // device_rank
  tmpSize = rank.size() * sizeof(int);
  cudaMalloc(&d_rank, tmpSize);
  cudaMemcpy(d_rank, rank.data(), tmpSize, cudaMemcpyHostToDevice);

  // device_Tree[ ].height
  int num_elements = Tree.size();
  vector<int> h_Tree_height(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    h_Tree_height[i] = Tree[i].height;
  }
  tmpSize = h_Tree_height.size() * sizeof(int);
  cudaMalloc(&d_Tree_height, tmpSize);
  cudaMemcpy(d_Tree_height, h_Tree_height.data(), tmpSize, cudaMemcpyHostToDevice);

  // Tree_posOptimize
  vector<int> h_Tree_posOptimize_flat;
  vector<int> h_Tree_posOptimize_idx;
  h_Tree_posOptimize_idx.push_back(0);
  for (int i = 0; i < Tree.size(); ++i) {
    for (int j = 0; j < Tree[i].posOptimize.size(); ++j) {
      h_Tree_posOptimize_flat.push_back(Tree[i].posOptimize[j]);
    }
    h_Tree_posOptimize_idx.push_back(h_Tree_posOptimize_flat.size());
  }
  tmpSize = h_Tree_posOptimize_flat.size() * sizeof(int);
  cudaMalloc(&d_Tree_posOptimize, tmpSize);
  cudaMemcpy(d_Tree_posOptimize, h_Tree_posOptimize_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_Tree_posOptimize_idx.size() * sizeof(int);
  cudaMalloc(&d_Tree_posOptimize_idx, tmpSize);
  cudaMemcpy(d_Tree_posOptimize_idx, h_Tree_posOptimize_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  // device_Tree[ ].Tree_dis
  vector<int> h_Tree_dis_flat;
  vector<int> h_Tree_dis_idx;
  h_Tree_dis_idx.push_back(0);
  for (int i = 0; i < Tree.size(); ++i) {
    for (int j = 0; j < Tree[i].dis.size(); ++j) {
      h_Tree_dis_flat.push_back(Tree[i].dis[j]);
    }
    h_Tree_dis_idx.push_back(h_Tree_dis_flat.size());
  }
  tmpSize = h_Tree_dis_flat.size() * sizeof(int);
  cudaMalloc(&d_Tree_dis, tmpSize);
  cudaMemcpy(d_Tree_dis, h_Tree_dis_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_Tree_dis_idx.size() * sizeof(int);
  cudaMalloc(&d_Tree_dis_idx, tmpSize);
  cudaMemcpy(d_Tree_dis_idx, h_Tree_dis_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  // device_Tree_branch
  num_elements = Tree.size();
  vector<uint8_t> h_Tree_branch(num_elements, false); // bool->uint8_t
  for (int i = 0; i < num_elements; ++i) {
    h_Tree_branch[i] = Tree[i].isBranchNode;
  }
  tmpSize = h_Tree_branch.size() * sizeof(bool);
  cudaMalloc(&d_Tree_branch, tmpSize);
  cudaMemcpy(d_Tree_branch, h_Tree_branch.data(), tmpSize, cudaMemcpyHostToDevice);

  // Tree_posP2H_3
  vector<int> h_Tree_posP2H_3_flat;

  vector<int> h_Tree_posP2H_3_idx1;
  h_Tree_posP2H_3_idx1.push_back(0);

  vector<int> h_Tree_posP2H_3_idx2;
  h_Tree_posP2H_3_idx2.push_back(0);

  for (int i = 0; i < Tree.size(); ++i) {
    for (int j = 0; j < Tree[i].posP2H_3.size(); ++j) {
      for (int k = 0; k < Tree[i].posP2H_3[j].size(); ++k) {
        h_Tree_posP2H_3_flat.push_back(Tree[i].posP2H_3[j][k]);
      }
      h_Tree_posP2H_3_idx1.push_back(h_Tree_posP2H_3_flat.size());
    }
    h_Tree_posP2H_3_idx2.push_back(h_Tree_posP2H_3_idx1.size() - 1);
  }
  tmpSize = h_Tree_posP2H_3_flat.size() * sizeof(int);
  cudaMalloc(&d_Tree_posP2H_3, tmpSize);
  cudaMemcpy(d_Tree_posP2H_3, h_Tree_posP2H_3_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_Tree_posP2H_3_idx1.size() * sizeof(int);
  cudaMalloc(&d_Tree_posP2H_3_idx1, tmpSize);
  cudaMemcpy(d_Tree_posP2H_3_idx1, h_Tree_posP2H_3_idx1.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_Tree_posP2H_3_idx2.size() * sizeof(int);
  cudaMalloc(&d_Tree_posP2H_3_idx2, tmpSize);
  cudaMemcpy(d_Tree_posP2H_3_idx2, h_Tree_posP2H_3_idx2.data(), tmpSize, cudaMemcpyHostToDevice);
}

void Graph::initGraphTravel() {
  /**
   * @brief graph travel
   * device_Neighbor(NNodeID) device_Neighbor(NWeight)    device_Neighbor_idx device_edge_vid_vid
   */
  h_Neighbor_idx.push_back(0);
  for (int i = 0; i < Neighbor.size(); ++i) {
    for (int j = 0; j < Neighbor[i].size(); ++j) {
      h_Neighbor_NodeID_flat.push_back(Neighbor[i][j].first);
      h_Neighbor_weight_flat.push_back(Neighbor[i][j].second);
      h_edge_vid_vid.push_back(i);
      h_edge_vid_vid.push_back(Neighbor[i][j].first);
    }
    h_Neighbor_idx.push_back(h_Neighbor_NodeID_flat.size());
  }
  int tmpSize = h_Neighbor_NodeID_flat.size() * sizeof(int);
  cudaMalloc(&d_Neighbor_NodeID, tmpSize);
  cudaMemcpy(d_Neighbor_NodeID, h_Neighbor_NodeID_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_Neighbor_weight_flat.size() * sizeof(int);
  cudaMalloc(&d_Neighbor_weight, tmpSize);
  cudaMemcpy(d_Neighbor_weight, h_Neighbor_weight_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_Neighbor_idx.size() * sizeof(int);
  cudaMalloc(&d_Neighbor_idx, tmpSize);
  cudaMemcpy(d_Neighbor_idx, h_Neighbor_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_edge_vid_vid.size() * sizeof(int);
  cudaMalloc(&d_edge_vid_vid, tmpSize);
  cudaMemcpy(d_edge_vid_vid, h_edge_vid_vid.data(), tmpSize, cudaMemcpyHostToDevice);
}

void Graph::freeMemory() {

  // moveDisIndexToDevice
  cudaFree(d_toRMQ);
  cudaFree(d_RMQIndex);
  cudaFree(d_RMQIndex_idx);
  cudaFree(d_rank);
  cudaFree(d_Tree_height);
  cudaFree(d_Tree_posOptimize);
  cudaFree(d_Tree_posOptimize_idx);
  cudaFree(d_Tree_dis);
  cudaFree(d_Tree_dis_idx);
  cudaFree(d_Tree_branch);
  cudaFree(d_Tree_posP2H_3);
  cudaFree(d_Tree_posP2H_3_idx1);
  cudaFree(d_Tree_posP2H_3_idx2);

  // initGraphTravel
  cudaFree(d_Neighbor_NodeID);
  cudaFree(d_Neighbor_weight);
  cudaFree(d_Neighbor_idx);
  cudaFree(d_edge_vid_vid);

  // brandesBCIndexGPU_5
  cudaFree(d_LCA);
  cudaFree(d_indegree);
  cudaFree(d_distance);
  cudaFree(d_successors_char);

  // for backPropagationKernel
  cudaFree(d_sigma);
  cudaFree(d_maxLayer);
  cudaFree(d_layer_start);
  cudaFree(d_queue_edge);
  cudaFree(d_delta_double);
  cudaFree(tmp_BC_double);
  cudaFree(d_BC_double);

  free(h_successors_char);
  free(h_indegree);
}

__global__ void BC_Kernel(const int *toRMQ, const int *RMQIndex, const int *RMQIndexIdx, const int *Tree_height,
                          int *_LCAQuery,
                          const int *rank, const int *Tree_posOptimize, const int *Tree_posOptimize_idx,
                          const bool *Tree_branch, const int *Tree_posP2H_3, const int *Tree_posP2H_3_idx1, const int *Tree_posP2H_3_idx2,
                          const int *Tree_dis, const int *Tree_dis_idx,
                          int *_distance,
                          const int *edge_vid_vid, const int *Neighbor_weight,
                          unsigned char *_successors, int *_indegree,
                          const int src, const int nodeNum, const int edgeNum, const int blockNum) {

  __shared__ int nodeOffset, edgeOffset;

  __shared__ int p1, x1;
  __shared__ int *LCAQuery;
  __shared__ int *distance;

  __shared__ unsigned char *successors;
  __shared__ int *indegree;
  for (int source = src + blockIdx.x; source < src + blockNum && source < nodeNum; source += gridDim.x) {

    // initialize
    if (threadIdx.x == 0) {

      nodeOffset = blockIdx.x * nodeNum;
      edgeOffset = blockIdx.x * edgeNum;

      LCAQuery = _LCAQuery + nodeOffset;
      distance = _distance + nodeOffset;

      int p2 = source;

      x1 = rank[p2];

      p1 = toRMQ[x1];

      successors = _successors + edgeOffset;
      indegree = _indegree + nodeOffset;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < nodeNum; idx += blockDim.x) {

      // 1.LCAQueryKernel

      int p = p1;
      int q = toRMQ[idx];
      if (p > q) {
        int x = p;
        p = q;
        q = x;
      }
      int len = q - p + 1;
      int i = 1, k = 0;
      while (i * 2 < len) {
        i *= 2;
        k++;
      }
      q = q - i + 1;

      int k_idx = RMQIndexIdx[k];
      int RMQIndex_k_p = RMQIndex[k_idx + p], RMQIndex_k_q = RMQIndex[k_idx + q];
      if (Tree_height[RMQIndex_k_p] < Tree_height[RMQIndex_k_q])
        LCAQuery[idx] = RMQIndex_k_p;
      else
        LCAQuery[idx] = RMQIndex_k_q;
    }
    __syncthreads();

    // 2.DistanceCountLayerQueryKernel
    for (int idx = threadIdx.x; idx < nodeNum; idx += blockDim.x) {
      if (idx == source) {
        distance[idx] = 0;
        continue;
      }
      int r1 = x1;
      int r2 = rank[idx];     // int r1 = rank[source],
      int LCA = LCAQuery[r2]; // int LCA = LCAQuery(r1, r2);
      int tmp = INF;

      if (LCA == r1) {
        int r1_idx = Tree_posOptimize[Tree_posOptimize_idx[r1 + 1] - 1];
        tmp = Tree_dis[Tree_dis_idx[r2] + r1_idx];

      } else if (LCA == r2) {
        int r2_idx = Tree_posOptimize[Tree_posOptimize_idx[r2 + 1] - 1];
        tmp = Tree_dis[Tree_dis_idx[r1] + r2_idx];

      } else if (Tree_branch[LCA]) {
        int sum = 0;

        int idx1 = Tree_posP2H_3_idx2[LCA];

        int minID = (Tree_posP2H_3_idx1[idx1 + source + 1] - Tree_posP2H_3_idx1[idx1 + source]) < (Tree_posP2H_3_idx1[idx1 + idx + 1] - Tree_posP2H_3_idx1[idx1 + idx]) ? source : idx;

        // vector<int> &pr = Tree[LCA].posP2H_3[minID];

        int r1_dis_start = Tree_dis_idx[r1];
        int r2_dis_start = Tree_dis_idx[r2];

        int loop = Tree_posP2H_3_idx1[idx1 + minID + 1] - Tree_posP2H_3_idx1[idx1 + minID];

        for (int i = loop - 1; i >= 0; --i) {

          int offset = Tree_posP2H_3_idx1[idx1 + minID] + i;

          sum = Tree_dis[r1_dis_start + Tree_posP2H_3[offset]] + Tree_dis[r2_dis_start + Tree_posP2H_3[offset]];

          if (tmp > sum)
            tmp = sum;
        }
      } else {
        int sum = 0;
        int r1_dis_start = Tree_dis_idx[r1];
        int r2_dis_start = Tree_dis_idx[r2];

        int loop = Tree_posOptimize_idx[LCA + 1] - Tree_posOptimize_idx[LCA];
        int LCA_posOptimize_start = Tree_posOptimize_idx[LCA];

        for (int i = loop - 1; i >= 0; --i) {

          int offset = Tree_posOptimize[LCA_posOptimize_start + i];
          sum = Tree_dis[r1_dis_start + offset] + Tree_dis[r2_dis_start + offset];

          if (tmp > sum)
            tmp = sum;
        }
      }
      distance[idx] = tmp;
    }
    __syncthreads();

    // 3. succ_out_pre_in_edge_Kernel && edgeLayerKernel
    for (int idx = threadIdx.x; idx < edgeNum; idx += blockDim.x) {
      int nodeID = edge_vid_vid[2 * idx], NNodeID = edge_vid_vid[2 * idx + 1], NWeight = Neighbor_weight[idx];

      if (distance[NNodeID] == distance[nodeID] + NWeight) {
        successors[idx] = true; // idx -> NNodeID
        atomicAdd(indegree + NNodeID, 1);
      } else {
        successors[idx] = false;
      }
    }
    __syncthreads();
  }
}

__global__ void backPropagationKernel(const int *edge_vid_vid,
                                      int *_count,
                                      const int *_maxLayer, int *_layer_start, int *_queue_edge,
                                      double *_delta, double *_tmp_BC,
                                      const int src, const int nodeNum, const int edgeNum, const int blockNum) {
  __shared__ int nodeOffset, edgeOffset;
  __shared__ int *count;
  __shared__ int maxLayer;
  __shared__ int *layer_start;
  __shared__ int *queue_edge;
  __shared__ double *delta, *tmp_BC;

  for (int source = src + blockIdx.x; source < src + blockNum && source < nodeNum; source += gridDim.x) {

    if (threadIdx.x == 0) {
      nodeOffset = blockIdx.x * nodeNum;
      edgeOffset = blockIdx.x * edgeNum;

      count = _count + nodeOffset;

      maxLayer = _maxLayer[source % blockNum];
      layer_start = _layer_start + nodeOffset;

      queue_edge = _queue_edge + edgeOffset;

      delta = _delta + nodeOffset;
      tmp_BC = _tmp_BC + nodeOffset;
    }
    __syncthreads();

    for (int i = maxLayer - 1; i >= 0; --i) {
      int layer_size = layer_start[i + 1] - layer_start[i];
      int layer_start_idx = layer_start[i];
      for (int idx = threadIdx.x; idx < layer_size; idx += blockDim.x) {
        int edgeID = queue_edge[layer_start_idx + idx];
        int nodeID = edge_vid_vid[edgeID * 2], NNodeID = edge_vid_vid[edgeID * 2 + 1];
        double tmp2 = count[nodeID] * (1 + delta[NNodeID]) / count[NNodeID];
        atomicAdd(delta + nodeID, tmp2);
      }
      __syncthreads();
    }

    // __syncthreads();

    // add BC
    for (int idx = threadIdx.x; idx < nodeNum; idx += blockDim.x) {
      tmp_BC[idx] = tmp_BC[idx] + delta[idx] * 0.5;
      delta[idx] = 0;
    }
    __syncthreads();
  }
}
__global__ void addKernel(double *tmp_BC,
                          double *BC,
                          const int blockNum,
                          const int nodeNum) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  for (; idx < nodeNum; idx += blockDim.x * gridDim.x) {

    double sum = 0.0;

    for (int i = 0; i < blockNum; i++) {
      sum += tmp_BC[idx + i * nodeNum];
    }

    BC[idx] += sum;
  }
}
void Graph::brandesBCIndexGPU_5(string roadName) {

  CHconsorderMT(orderFile);
  makeTree();
  makeIndex();

  buildIndexMethod = "P2H";
  if (buildIndexMethod == "optimizeVS") {
    optimizeVS();
  } else if (buildIndexMethod == "P2H") {
    optimizeVS();
    vector<int> ress;
    int k = 10, d = 5, R = 1000000;
    SVSP_P2H(k, d, R, ress);
  }

  moveDisIndexToDevice();
  initGraphTravel();

  // allocateDeviceMemory
  cudaMalloc(&d_LCA, nodenum * blockNum * sizeof(int));
  cudaMalloc(&d_indegree, nodenum * blockNum * sizeof(int));
  cudaMalloc(&d_distance, nodenum * blockNum * sizeof(int));
  cudaMalloc(&d_successors_char, edgenum * blockNum * sizeof(unsigned char));

  // for backPropagationKernel
  cudaMalloc(&d_sigma, blockNum * nodenum * sizeof(int));
  cudaMalloc(&d_maxLayer, nodenum * sizeof(int));
  cudaMalloc(&d_layer_start, blockNum * nodenum * sizeof(int));

  cudaMalloc(&d_queue_edge, blockNum * edgenum * sizeof(int));
  cudaMalloc(&d_delta_double, blockNum * nodenum * sizeof(double));
  cudaMalloc(&tmp_BC_double, blockNum * nodenum * sizeof(double));
  cudaMalloc(&d_BC_double, nodenum * sizeof(double));

  cudaMemset(d_delta_double, 0, blockNum * nodenum * sizeof(double));
  cudaMemset(tmp_BC_double, 0, blockNum * nodenum * sizeof(double));
  cudaMemset(d_BC_double, 0, nodenum * sizeof(double));

  // allocateHostMemory
  h_successors_char = (unsigned char *)malloc(edgenum * blockNum * sizeof(unsigned char));
  h_indegree = (int *)malloc(nodenum * blockNum * sizeof(int));

  int *sigmaVec = (int *)malloc(blockNum * nodenum * sizeof(int));
  int *h_maxLayer_vec = (int *)malloc(blockNum * sizeof(int));
  int *h_queue_edge_vec = (int *)malloc(blockNum * edgenum * sizeof(int));
  int *h_layer_start_vec = (int *)malloc(blockNum * nodenum * sizeof(int));

  vector<vector<int>> h_queue_vec(blockNum, vector<int>(nodenum));
  vector<vector<int>> h_queue_edge_idx_vec(blockNum);

  for (int s = 0; s < nodenum; s += blockNum) {

    cudaMemset(d_indegree, 0, nodenum * blockNum * sizeof(int));

    BC_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_toRMQ, d_RMQIndex, d_RMQIndex_idx, d_Tree_height, d_LCA, d_rank, d_Tree_posOptimize, d_Tree_posOptimize_idx, d_Tree_branch, d_Tree_posP2H_3, d_Tree_posP2H_3_idx1, d_Tree_posP2H_3_idx2, d_Tree_dis, d_Tree_dis_idx, d_distance, d_edge_vid_vid, d_Neighbor_weight, d_successors_char, d_indegree, s, nodenum, edgenum, blockNum);
    cudaDeviceSynchronize();

    cudaMemcpy(h_indegree, d_indegree, nodenum * blockNum * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_successors_char, d_successors_char, edgenum * blockNum * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    int end = s + blockNum;
    if (end > nodenum)
      end = nodenum;

#pragma omp parallel for num_threads(numOfThread)
    for (int source = s; source < end; source++) {

      int baseOffset = source % blockNum;
      int nodeOffset = baseOffset * nodenum, edgeOffset = baseOffset * edgenum;

      int *indegree_tmp = h_indegree + nodeOffset;
      unsigned char *successors_tmp = h_successors_char + edgeOffset;

      int *sigma_tmp = sigmaVec + nodeOffset;
      auto &h_queue_tmp = h_queue_vec[baseOffset];
      int *h_queue_edge_tmp = h_queue_edge_vec + edgeOffset;
      auto &h_queue_edge_idx_tmp = h_queue_edge_idx_vec[baseOffset];

      int *layer_start_tmp = h_layer_start_vec + baseOffset * nodenum;

      memset(sigma_tmp, 0, nodenum * sizeof(int));
      sigma_tmp[source] = 1;

      int layerQueueStart = 0, layerSize = 1, h_cnt = 0, totalSize_edge = 0, h_cnt_edge = 0;

      h_queue_edge_idx_tmp.clear();
      h_queue_edge_idx_tmp.push_back(0);
      h_queue_tmp[0] = source;

      for (int j = h_Neighbor_idx[source]; j < h_Neighbor_idx[source + 1]; ++j) {
        if (successors_tmp[j] == true) {
          int NNodeID = h_Neighbor_NodeID_flat[j];
          sigma_tmp[NNodeID] += sigma_tmp[source];
          --indegree_tmp[NNodeID];
          if (indegree_tmp[NNodeID] == 0) {
            h_queue_tmp[1 + h_cnt] = NNodeID;
            ++h_cnt;
          }
        }
      }

      layerQueueStart = 1;
      layerSize = h_cnt;

      while (layerQueueStart + layerSize != nodenum) {

        h_cnt = 0;
        h_cnt_edge = 0;
        for (int idx = 0; idx < layerSize; ++idx) {
          int nodeID = h_queue_tmp[idx + layerQueueStart];
          for (int j = h_Neighbor_idx[nodeID]; j < h_Neighbor_idx[nodeID + 1]; ++j) {
            if (successors_tmp[j] == true) {
              int NNodeID = h_Neighbor_NodeID_flat[j];
              sigma_tmp[NNodeID] += sigma_tmp[nodeID];
              --indegree_tmp[NNodeID];
              if (indegree_tmp[NNodeID] == 0) {
                h_queue_tmp[layerQueueStart + layerSize + h_cnt] = NNodeID;
                ++h_cnt;
              }
              h_queue_edge_tmp[totalSize_edge + h_cnt_edge] = j;
              h_cnt_edge++;
            }
          }
        }
        layerQueueStart = layerQueueStart + layerSize;
        layerSize = h_cnt;
        totalSize_edge += h_cnt_edge;
        h_queue_edge_idx_tmp.push_back(totalSize_edge);
      }

      h_maxLayer_vec[source % blockNum] = h_queue_edge_idx_tmp.size() - 1;
      for (int i = 0; i < h_queue_edge_idx_tmp.size(); i++) {
        layer_start_tmp[i] = h_queue_edge_idx_tmp[i];
      }
    }

    cudaMemcpy(d_sigma, sigmaVec, blockNum * nodenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxLayer, h_maxLayer_vec, blockNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue_edge, h_queue_edge_vec, edgenum * blockNum * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < blockNum; i++) {
      cudaMemcpy(d_layer_start + i * nodenum, h_layer_start_vec + i * nodenum, (h_maxLayer_vec[i] + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    //  backPropagationKernel
    backPropagationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_edge_vid_vid,
                                                              d_sigma,
                                                              d_maxLayer, d_layer_start, d_queue_edge,
                                                              d_delta_double, tmp_BC_double,
                                                              s, nodenum, edgenum, blockNum);
    cudaDeviceSynchronize();
  }

  addKernel<<<blocksPerGrid, threadsPerBlock>>>(tmp_BC_double, d_BC_double, blockNum, nodenum);
  cudaDeviceSynchronize();
  BC.resize(nodenum);
  cudaMemcpy(BC.data(), d_BC_double, nodenum * sizeof(double), cudaMemcpyDeviceToHost);

  freeMemory();
  free(sigmaVec);
  free(h_maxLayer_vec);
  free(h_queue_edge_vec);
  free(h_layer_start_vec);
}

/**
nvcc -std=c++17 -O3 -arch=sm_86  -ccbin /usr/bin/g++-10   -Xcompiler -fopenmp QBC.cu  src/Graph.cpp  src/buildIndex.cpp  -o QBC

./QBC data/roadData/LosAngeles.txt data/orderFile/LosAngelesOrder.txt 0 LosAngelesBC.txt
./QBC data/roadData/bay.txt data/orderFile/bayOrder.txt 0 bayBC.txt

*/
int main(int argc, char *argv[]) {

  cout.precision(8);
  cout << fixed;

  auto t11 = std::chrono::high_resolution_clock::now(), t22 = t11;
  roadDataPath = argv[1];
  orderFile = argv[2];
  int device = stoi(argv[3]);
  cudaSetDevice(device);

  Graph g;
  g.ReadGraph(roadDataPath);
  cout << "roadDataPath:" << roadDataPath << "\t" << g.nodenum << "\t" << g.edgenum << "\n";

  t11 = std::chrono::high_resolution_clock::now();
  g.brandesBCIndexGPU_5(roadDataPath);
  t22 = std::chrono::high_resolution_clock::now();
  double bcTime = std::chrono::duration_cast<std::chrono::duration<double>>(t22 - t11).count();
  cout << bcTime << "  seconds (---totalTime---)" << endl;

  if (argc >= 5) {
    string writeFile = argv[4];
    ofstream ofs;
    ofs.precision(2);
    ofs.setf(ios::fixed, ios::floatfield);
    ofs.open(writeFile, ios::out);
    ofs << "vertexID  BCValue" << endl;
    for (int i = 0; i < g.nodenum; ++i) {
      ofs << i + 1 << " " << g.BC[i] << endl;
    }
    ofs.close();
    cout << "The BC value has been save: " << writeFile << endl;
  }

  if (argc >= 6) {
    // check the BC value
    ifstream ifs;
    ifs.open(argv[5], ios::in);
    int nodenum, edgenum, vid;
    double bc_value;
    ifs >> nodenum >> edgenum;
    vector<double> standardBC(nodenum);
    while (ifs >> vid >> bc_value) {
      standardBC[vid - 1] = bc_value;
    }
    for (int i = 0; i < nodenum; i++) {
      if (abs(g.BC[i] - standardBC[i]) > 1e-1) {
        cout << "BC value is not correct for node " << i << "\t" << g.BC[i] << "\t" << standardBC[i] << endl;
        exit(1);
      }
    }
    cout << "BC value is correct\n\n"
         << endl;
  }
}
