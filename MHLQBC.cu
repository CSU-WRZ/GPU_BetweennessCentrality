#include "include/Graph.h"

const int threadsPerBlock = 256, blocksPerGrid = 256, blockNum = 256;
string roadDataPath;

void Graph::moveCountIndexToDevice(TreeDecomp &td) {

  int tmpSize = 0;
  // copy rank_
  tmpSize = (nodenum + 1) * sizeof(int);
  cudaMalloc(&d_rank_, tmpSize);
  cudaMemcpy(d_rank_, td.rank_, tmpSize, cudaMemcpyHostToDevice);

  // copy euler_tour_pos_
  tmpSize = (nodenum + 1) * sizeof(int);
  cudaMalloc(&d_euler_tour_pos_, tmpSize);
  cudaMemcpy(d_euler_tour_pos_, td.euler_tour_pos_, tmpSize, cudaMemcpyHostToDevice);

  // copy rmq_index_
  vector<int> h_RMQIndex_flat;
  vector<int> h_RMQIndex_flat_idx;
  h_RMQIndex_flat_idx.push_back(0);
  for (int i = 0; i < td.rmq_index_.size(); ++i) {
    for (int j = 0; j < td.rmq_index_[i].size(); ++j) {
      h_RMQIndex_flat.push_back(td.rmq_index_[i][j]);
    }
    h_RMQIndex_flat_idx.push_back(h_RMQIndex_flat.size());
  }
  tmpSize = h_RMQIndex_flat.size() * sizeof(int);
  cudaMalloc(&d_rmq_index_, tmpSize);
  cudaMemcpy(d_rmq_index_, h_RMQIndex_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_RMQIndex_flat_idx.size() * sizeof(int);
  cudaMalloc(&d_rmq_index_idx, tmpSize);
  cudaMemcpy(d_rmq_index_idx, h_RMQIndex_flat_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  // copy Tree[ ].height
  int num_elements = td.tree_.size();
  vector<int> h_Tree_height(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    h_Tree_height[i] = td.tree_[i].height;
  }
  tmpSize = h_Tree_height.size() * sizeof(int);
  cudaMalloc(&d_tree_height, tmpSize);
  cudaMemcpy(d_tree_height, h_Tree_height.data(), tmpSize, cudaMemcpyHostToDevice);

  // copy tree_[lca].pos.size()
  num_elements = td.tree_.size();
  vector<int> h_tree_posSize(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    h_tree_posSize[i] = td.tree_[i].pos.size();
  }
  tmpSize = h_tree_posSize.size() * sizeof(int);
  cudaMalloc(&d_tree_posSize, tmpSize);
  cudaMemcpy(d_tree_posSize, h_tree_posSize.data(), tmpSize, cudaMemcpyHostToDevice);

  // copy tree_[].pos.
  vector<int> h_tree_pos_flat;
  vector<int> h_tree_pos_idx;
  h_tree_pos_idx.push_back(0);
  for (int i = 0; i < td.tree_.size(); ++i) {
    for (int j = 0; j < td.tree_[i].pos.size(); ++j) {
      h_tree_pos_flat.push_back(td.tree_[i].pos[j]);
    }
    h_tree_pos_idx.push_back(h_tree_pos_flat.size());
  }
  tmpSize = h_tree_pos_flat.size() * sizeof(int);
  cudaMalloc(&d_tree_pos, tmpSize);
  cudaMemcpy(d_tree_pos, h_tree_pos_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_tree_pos_idx.size() * sizeof(int);
  cudaMalloc(&d_tree_pos_idx, tmpSize);
  cudaMemcpy(d_tree_pos_idx, h_tree_pos_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  // copy tree_[].dis.
  vector<int> h_tree_dis_flat;
  vector<int> h_tree_dis_idx;
  h_tree_dis_idx.push_back(0);
  for (int i = 0; i < td.tree_.size(); ++i) {
    for (int j = 0; j < td.tree_[i].dis.size(); ++j) {
      h_tree_dis_flat.push_back(td.tree_[i].dis[j]);
    }
    h_tree_dis_idx.push_back(h_tree_dis_flat.size());
  }
  tmpSize = h_tree_dis_flat.size() * sizeof(int);
  cudaMalloc(&d_tree_dis, tmpSize);
  cudaMemcpy(d_tree_dis, h_tree_dis_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_tree_dis_idx.size() * sizeof(int);
  cudaMalloc(&d_tree_dis_idx, tmpSize);
  cudaMemcpy(d_tree_dis_idx, h_tree_dis_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  // copy tree_[].cnt.
  vector<int> h_tree_cnt_flat;
  vector<int> h_tree_cnt_idx;
  h_tree_cnt_idx.push_back(0);
  for (int i = 0; i < td.tree_.size(); ++i) {
    for (int j = 0; j < td.tree_[i].cnt.size(); ++j) {
      h_tree_cnt_flat.push_back(td.tree_[i].cnt[j]);
    }
    h_tree_cnt_idx.push_back(h_tree_cnt_flat.size());
  }
  tmpSize = h_tree_cnt_flat.size() * sizeof(int);
  cudaMalloc(&d_tree_cnt, tmpSize);
  cudaMemcpy(d_tree_cnt, h_tree_cnt_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_tree_cnt_idx.size() * sizeof(int);
  cudaMalloc(&d_tree_cnt_idx, tmpSize);
  cudaMemcpy(d_tree_cnt_idx, h_tree_cnt_idx.data(), tmpSize, cudaMemcpyHostToDevice);

  // copy tree_[].my_cnt.
  vector<int> h_my_tree_cnt_flat;
  vector<int> h_my_tree_cnt_idx;
  h_my_tree_cnt_idx.push_back(0);
  for (int i = 0; i < td.tree_.size(); ++i) {
    for (int j = 0; j < td.tree_[i].my_cnt.size(); ++j) {
      h_my_tree_cnt_flat.push_back(td.tree_[i].my_cnt[j]);
    }
    h_my_tree_cnt_idx.push_back(h_my_tree_cnt_flat.size());
  }
  tmpSize = h_my_tree_cnt_flat.size() * sizeof(int);
  cudaMalloc(&d_my_tree_cnt, tmpSize);
  cudaMemcpy(d_my_tree_cnt, h_my_tree_cnt_flat.data(), tmpSize, cudaMemcpyHostToDevice);

  tmpSize = h_my_tree_cnt_idx.size() * sizeof(int);
  cudaMalloc(&d_my_tree_cnt_idx, tmpSize);
  cudaMemcpy(d_my_tree_cnt_idx, h_my_tree_cnt_idx.data(), tmpSize, cudaMemcpyHostToDevice);
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

void Graph::allocateDeviceMemory() {

  // 1.LCAQueryKernel
  cudaMalloc(&d_LCAQuery, blockNum * (nodenum) * sizeof(int));

  // 2.DistanceCountLayerQueryKernel

  cudaMalloc(&d_distance, blockNum * nodenum * sizeof(int));
  cudaMalloc(&d_sigma, blockNum * nodenum * sizeof(int));
  cudaMalloc(&d_layer, blockNum * nodenum * sizeof(int));
  cudaMalloc(&d_maxLayer, nodenum * sizeof(int));
  cudaMemset(d_maxLayer, 0, nodenum * sizeof(int));

  // 3.succ_out_pre_in_edge_Kernel
  cudaMalloc(&d_successors_char, blockNum * edgenum * sizeof(unsigned char));

  // 4.edgeLayerKernel
  cudaMalloc(&d_layer_edge, blockNum * edgenum * sizeof(int));
  cudaMalloc(&d_layer_size, blockNum * nodenum * sizeof(int));
  cudaMemset(d_layer_size, 0, blockNum * nodenum * sizeof(int));

  // 5.edgeLayerKernel
  cudaMalloc(&d_queue_edge, blockNum * edgenum * sizeof(int));
  cudaMalloc(&d_layer_start, blockNum * nodenum * sizeof(int));

  cudaMalloc(&d_delta_double, blockNum * nodenum * sizeof(double));
  cudaMemset(d_delta_double, 0, blockNum * nodenum * sizeof(double));

  cudaMalloc(&tmp_BC_double, blockNum * nodenum * sizeof(double));
  cudaMemset(tmp_BC_double, 0, blockNum * nodenum * sizeof(double));

  cudaMalloc(&d_BC_double, nodenum * sizeof(double));
  cudaMemset(d_BC_double, 0, nodenum * sizeof(double));
}

void Graph::freeMemory() {

  // moveCountIndexToDevice
  cudaFree(d_rank_);
  cudaFree(d_euler_tour_pos_);
  cudaFree(d_rmq_index_);
  cudaFree(d_rmq_index_idx);
  cudaFree(d_tree_height);
  cudaFree(d_tree_posSize);
  cudaFree(d_tree_pos);
  cudaFree(d_tree_pos_idx);
  cudaFree(d_tree_dis);
  cudaFree(d_tree_dis_idx);
  cudaFree(d_tree_cnt);
  cudaFree(d_tree_cnt_idx);
  cudaFree(d_my_tree_cnt);
  cudaFree(d_my_tree_cnt_idx);

  // initGraphTravel
  cudaFree(d_Neighbor_NodeID);
  cudaFree(d_Neighbor_weight);
  cudaFree(d_Neighbor_idx);
  cudaFree(d_edge_vid_vid);

  // allocateDeviceMemory
  cudaFree(d_LCAQuery);
  cudaFree(d_distance);
  cudaFree(d_sigma);
  cudaFree(d_layer);
  cudaFree(d_maxLayer);
  cudaFree(d_successors_char);
  cudaFree(d_layer_edge);
  cudaFree(d_layer_size);
  cudaFree(d_queue_edge);
  cudaFree(d_layer_start);
  cudaFree(d_delta_double);
  cudaFree(tmp_BC_double);
  cudaFree(d_BC_double);
}

__global__ void BC_Kernel(const int *euler_tour_pos_,
                          const int *rmq_index_,
                          const int *rmq_index_idx,
                          const int *tree_height,
                          const int *rank_,
                          int *_LCAQuery,

                          const int *tree_posSize,
                          const int *tree_pos, const int *tree_pos_idx,
                          const int *tree_dis, const int *tree_dis_idx,
                          const int *tree_cnt, const int *tree_cnt_idx,
                          const int *my_tree_cnt, const int *my_tree_cnt_idx,
                          int *_distance, int *_count, int *_layer, int *_maxLayer,

                          unsigned char *_successors, int *_layer_edge, int *_layer_size,
                          int *_layer_start, int *_queue_edge,
                          double *_delta, double *_tmp_BC,
                          const int *edge_vid_vid, const int *Neighbor_weight, const int nodeNum, const int edgeNum, const int blockNum) {

  __shared__ int nodeOffset, edgeOffset;

  __shared__ int p1, x1;
  __shared__ int *LCAQuery;
  __shared__ int *distance, *count, *layer, *maxLayer;

  __shared__ unsigned char *successors;
  __shared__ int *layer_edge, *layer_size, *layer_start;
  __shared__ int *queue_edge;
  __shared__ double *delta, *tmp_BC;
  __shared__ int tmp_back;
  for (int source = blockIdx.x; source < nodeNum; source += gridDim.x) {

    if (threadIdx.x == 0) {
      // initialize

      nodeOffset = blockIdx.x * nodeNum;
      edgeOffset = blockIdx.x * edgeNum;

      LCAQuery = _LCAQuery + nodeOffset;
      distance = _distance + nodeOffset;
      count = _count + nodeOffset;
      layer = _layer + nodeOffset;
      maxLayer = _maxLayer + source;

      int p2 = source + 1;

      x1 = rank_[p2];

      p1 = euler_tour_pos_[x1];

      successors = _successors + edgeOffset;
      layer_edge = _layer_edge + edgeOffset;

      layer_size = _layer_size + nodeOffset;
      layer_start = _layer_start + nodeOffset;

      queue_edge = _queue_edge + edgeOffset;

      tmp_back = edgeNum - 1;

      delta = _delta + nodeOffset;
      tmp_BC = _tmp_BC + nodeOffset;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < nodeNum; idx += blockDim.x) {

      // 1.LCAQueryKernel
      int p = p1;
      int q = euler_tour_pos_[idx];
      if (p > q) {
        int tmp = p;
        p = q;
        q = tmp;
      }
      int len = q - p + 1;

      // int i = 1, k = 0;
      // while (i * 2 < len) {
      //   i *= 2;
      //   k++;
      // }

      int k = 31 - __clz(len); //  floor(log2(len))
      int i = 1 << k;

      q = q - i + 1;

      int k_idx = rmq_index_idx[k];
      int RMQIndex_k_p = rmq_index_[k_idx + p], RMQIndex_k_q = rmq_index_[k_idx + q];
      if (tree_height[RMQIndex_k_p] < tree_height[RMQIndex_k_q])
        LCAQuery[idx] = RMQIndex_k_p;
      else
        LCAQuery[idx] = RMQIndex_k_q;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < nodeNum; idx += blockDim.x) {

      // 2.DistanceCountLayerQueryKernel
      int p = source + 1;
      int q = idx + 1;
      int x = x1;
      int ret = 0, my_ret = 0;
      if (p == q) {     // maybe can test remove+1
        count[idx] = 1; // 1 or 0 ？
        distance[idx] = 0;
        layer[idx] = 0;
      } else {
        int y = rank_[q];      // int r1 = rank[source],
        int lca = LCAQuery[y]; // int LCA = LCAQuery(r1, r2);
        int dis = 999999999;
        int cnt = 0;
        int my_cnt = 0;

        int ps = tree_posSize[lca];
        // int position = pos_[lca][pos_size_[lca] - 1];
        int position = tree_pos[tree_pos_idx[lca] + ps - 1];

        if (lca == x || lca == y) {
          if (lca == y) {
            int v = y;
            y = x;
            x = v;
            v = p;
            p = q;
            q = v;
          }
          int a = tree_posSize[x] - 1;
          int pos = tree_pos[tree_pos_idx[x] + a];
          dis = tree_dis[tree_dis_idx[y] + pos];

          ret = tree_cnt[tree_cnt_idx[y] + pos];          // count[idx] = tree_cnt[tree_cnt_idx[y] + pos];
          my_ret = my_tree_cnt[my_tree_cnt_idx[y] + pos]; // count[idx] = tree_cnt[tree_cnt_idx[y] + pos];

          position--;
        }
        int tmp_dis = 999999999;
        int dx = tree_dis_idx[x], dy = tree_dis_idx[y];
        int cx = tree_cnt_idx[x], cy = tree_cnt_idx[y];
        int my_cx = my_tree_cnt_idx[x], my_cy = my_tree_cnt_idx[y];

        for (int i = 0; i <= position; i++) {
          int tmp = tree_dis[dx + i] + tree_dis[dy + i];
          if (tmp_dis > tmp) {
            tmp_dis = tmp;
            cnt = tree_cnt[cx + i] * tree_cnt[cy + i];
            my_cnt = my_tree_cnt[my_cx + i] + my_tree_cnt[my_cy + i];

          } else if (tmp_dis == tmp) {
            cnt += tree_cnt[cx + i] * tree_cnt[cy + i];
            if (my_tree_cnt[my_cx + i] + my_tree_cnt[my_cy + i] > my_cnt) {
              my_cnt = my_tree_cnt[my_cx + i] + my_tree_cnt[my_cy + i];
            }
          }
        }

        if (dis > tmp_dis) {
          dis = tmp_dis;
          ret = cnt; // count[idx] = cnt;
          my_ret = my_cnt;

        } else if (dis == tmp_dis) {
          ret += cnt; // count[idx] += cnt;
          if (my_cnt > my_ret) {
            my_ret = my_cnt;
          }
        }

        atomicMax(maxLayer, my_ret);

        layer[idx] = my_ret;
        count[idx] = ret;
        distance[idx] = dis;
      }
    }
    __syncthreads();

    // 3. succ_out_pre_in_edge_Kernel && edgeLayerKernel
    for (int idx = threadIdx.x; idx < edgeNum; idx += blockDim.x) {
      int nodeID = edge_vid_vid[2 * idx], NNodeID = edge_vid_vid[2 * idx + 1], NWeight = Neighbor_weight[idx];
      if (distance[NNodeID] == distance[nodeID] + NWeight) {
        successors[idx] = true; // idx -> NNodeID

        if (nodeID != source) {
          layer_edge[idx] = layer[NNodeID];
          atomicAdd(layer_size + layer_edge[idx], 1);
        } else {
          layer_edge[idx] = 1;
        }
      } else {
        successors[idx] = false;
        layer_edge[idx] = 1;
      }
    }
    __syncthreads();

    // 4. compute layer_start
    if (threadIdx.x == 0) {
      layer_start[*maxLayer] = edgeNum - layer_size[*maxLayer];
      for (int i = *maxLayer - 1; i >= 2; --i) {
        layer_start[i] = layer_start[i + 1] - layer_size[i];
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < edgeNum; idx += blockDim.x) {
      int layer = layer_edge[idx];
      if (layer > 1) {
        int old = atomicAdd(layer_start + layer, 1);
        queue_edge[old] = idx;
      }
    }
    __syncthreads();

    for (int i = *maxLayer; i > 1; --i) {
      for (int idx = threadIdx.x; idx < layer_size[i]; idx += blockDim.x) {
        int edgeID = queue_edge[tmp_back - idx];
        int nodeID = edge_vid_vid[edgeID * 2], NNodeID = edge_vid_vid[edgeID * 2 + 1];

        double tmp2 = count[nodeID] * (1 + delta[NNodeID]) / count[NNodeID];
        atomicAdd(delta + nodeID, tmp2);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        tmp_back -= layer_size[i];
        layer_size[i] = 0;
      }
      __syncthreads();
    }

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

  string graph_filename = roadDataPath, index_filename = "tmpRoadCountIndex";
  TreeDecomp td(graph_filename.c_str(), index_filename.c_str());
  td.Reduce();
  td.MakeTree();
  td.MakeIndex();

  // auto t1 = std::chrono::high_resolution_clock::now();
  moveCountIndexToDevice(td);
  initGraphTravel();
  allocateDeviceMemory();

  BC_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_euler_tour_pos_,
                                                d_rmq_index_,
                                                d_rmq_index_idx,
                                                d_tree_height,
                                                d_rank_,
                                                d_LCAQuery,

                                                d_tree_posSize,
                                                d_tree_pos, d_tree_pos_idx,
                                                d_tree_dis, d_tree_dis_idx,
                                                d_tree_cnt, d_tree_cnt_idx,
                                                d_my_tree_cnt, d_my_tree_cnt_idx,
                                                d_distance, d_sigma, d_layer, d_maxLayer,

                                                d_successors_char, d_layer_edge, d_layer_size, d_layer_start, d_queue_edge,

                                                d_delta_double, tmp_BC_double, d_edge_vid_vid, d_Neighbor_weight, nodenum, edgenum, blockNum);

  cudaDeviceSynchronize();

  addKernel<<<blocksPerGrid, threadsPerBlock>>>(tmp_BC_double, d_BC_double, blockNum, nodenum);
  cudaDeviceSynchronize();

  BC.resize(nodenum);
  cudaMemcpy(BC.data(), d_BC_double, nodenum * sizeof(double), cudaMemcpyDeviceToHost);

  // auto t2 = std::chrono::high_resolution_clock::now();
  // cout << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << "  seconds (---main kernel---)" << endl;
  freeMemory();
}

/*
nvcc -std=c++17 -O3 -arch=sm_86  -ccbin /usr/bin/g++-10  MHLQBC.cu  src/TreeCount/TreeDecomp.cpp src/TreeCount/Graph.cpp src/Graph.cpp src/buildIndex.cpp  -o MHLQBC

./MHLQBC data/roadData/LosAngeles.txt 0 LosAngelesBC.txt
./MHLQBC data/roadData/bay.txt 0 bayBC.txt

*/

int main(int argc, char *argv[]) {

  cout.precision(8);
  cout << fixed;

  auto t11 = std::chrono::high_resolution_clock::now(), t22 = t11;
  roadDataPath = argv[1];
  int device = stoi(argv[2]);
  cudaSetDevice(device); // deviceID

  Graph g;
  g.ReadGraph(roadDataPath);
  cout << "roadDataPath:" << roadDataPath << "\t" << g.nodenum << "\t" << g.edgenum << "\n";

  t11 = std::chrono::high_resolution_clock::now();
  g.brandesBCIndexGPU_5(roadDataPath);
  t22 = std::chrono::high_resolution_clock::now();
  double bcTime = std::chrono::duration_cast<std::chrono::duration<double>>(t22 - t11).count();
  cout << bcTime << "  seconds (---totalTime---)" << endl;

  if (argc >= 4) {
    string writeFile = argv[3];
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

  /*     if (argc >= 5) {
        // check the BC value
        ifstream ifs;
        ifs.open(argv[4], ios::in);
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
      }  */
}
