#include "include/Graph.h"
#include <iomanip>
const int threadsPerBlock = 256, blocksPerGrid = 256, blockNum = 256;
string roadDataPath, queryBatchPath;

void processBactch(int nodenum, int &batckSize, vector<int> &randNumber);

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

  cudaMalloc(&d_LCAQuery, blockNum * nodenum * sizeof(int));
  cudaMalloc(&d_distance, blockNum * nodenum * sizeof(int));
  cudaMalloc(&d_sigma, blockNum * nodenum * sizeof(int));
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
}

__global__ void LCAQueryKernel(const int *euler_tour_pos_, const int *rmq_index_, const int *rmq_index_idx, const int *tree_height, int *LCAQuery, int p, const int nodeNum) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x; //_q,r2 = rank[ID2]
  if (idx < nodeNum) {
    int q = euler_tour_pos_[idx];
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

    int k_idx = rmq_index_idx[k];
    int RMQIndex_k_p = rmq_index_[k_idx + p], RMQIndex_k_q = rmq_index_[k_idx + q];
    if (tree_height[RMQIndex_k_p] < tree_height[RMQIndex_k_q])
      LCAQuery[idx] = RMQIndex_k_p;
    else
      LCAQuery[idx] = RMQIndex_k_q;
  }
}

__global__ void DistanceCountQueryKernel(const int *rank_, const int *LCAQuery, const int *tree_posSize, const int *tree_pos, const int *tree_pos_idx,
                                         const int *tree_dis, const int *tree_dis_idx, const int *tree_cnt, const int *tree_cnt_idx,
                                         int *distance, int *count, int p, int x, const int nodeNum) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x; // ID2 ,idx <-> q+1

  if (idx < nodeNum) {
    int q = idx + 1, ret = 0;
    if (p == q) {     // maybe can test remove+1
      count[idx] = 1; // 1 or 0 ？
      distance[idx] = 0;
      return;
    }

    int y = rank_[q];      // int r1 = rank[source],
    int lca = LCAQuery[y]; // int LCA = LCAQuery(r1, r2);
    int dis = 999999999;
    int cnt = 0;
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

      ret = tree_cnt[tree_cnt_idx[y] + pos]; // count[idx] = tree_cnt[tree_cnt_idx[y] + pos];

      position--;
    }
    int tmp_dis = 999999999;
    int dx = tree_dis_idx[x], dy = tree_dis_idx[y];
    int cx = tree_cnt_idx[x], cy = tree_cnt_idx[y];
    for (int i = 0; i <= position; i++) {
      int tmp = tree_dis[dx + i] + tree_dis[dy + i];
      if (tmp_dis > tmp) {
        tmp_dis = tmp;
        cnt = tree_cnt[cx + i] * tree_cnt[cy + i];

      } else if (tmp_dis == tmp) {
        cnt += tree_cnt[cx + i] * tree_cnt[cy + i];
      }
    }

    if (dis > tmp_dis) {
      dis = tmp_dis;
      ret = cnt; // count[idx] = cnt;
    } else if (dis == tmp_dis) {
      ret += cnt; // count[idx] += cnt;
    }

    count[idx] = ret;
    distance[idx] = dis;
  }
}

__global__ void sourceDependencyKernel2(const int *distance, const int *sigma, const int *point_distance, const int *point_sigma, double *d_PointBC, const int s, const int *point, const int querySize, const int nodeNum) {
  int t = threadIdx.x + blockIdx.x * blockDim.x + s + 1;
  if (t < nodeNum) {
    double sigma_t = 1.0 / sigma[t];
    int distance_t = distance[t];
    for (int i = 0; i < querySize; ++i) {
      if (s != point[i] && t != point[i] && distance_t == point_distance[i * nodeNum + s] + point_distance[i * nodeNum + t]) {
        double tmp = point_sigma[i * nodeNum + s] * point_sigma[i * nodeNum + t] * sigma_t;
        atomicAdd(d_PointBC + i, tmp);
      }
    }
  }
}

__global__ void BC_Kernel_merge(const int *euler_tour_pos_,
                                const int *rmq_index_,
                                const int *rmq_index_idx,
                                const int *tree_height,
                                const int *rank_,
                                int *_LCAQuery,

                                const int *tree_posSize,
                                const int *tree_pos, const int *tree_pos_idx,
                                const int *tree_dis, const int *tree_dis_idx,
                                const int *tree_cnt, const int *tree_cnt_idx,
                                int *_distance, int *_count,

                                const int *edge_vid_vid, const int *Neighbor_weight, const int nodeNum, const int edgeNum, const int blockNum,

                                const int *point_distance, const int *point_sigma, double *_d_PointBC, const int *point, const int querySize

) {

  __shared__ int nodeOffset;

  __shared__ int p1, x1;
  __shared__ int *LCAQuery;
  __shared__ int *distance, *count;
  __shared__ double *d_PointBC;

  for (int source = blockIdx.x; source < nodeNum; source += gridDim.x) {

    // initialize
    if (threadIdx.x == 0) {

      nodeOffset = blockIdx.x * nodeNum;

      LCAQuery = _LCAQuery + nodeOffset;
      distance = _distance + nodeOffset;
      count = _count + nodeOffset;

      int p2 = source + 1;

      x1 = rank_[p2];

      p1 = euler_tour_pos_[x1];

      d_PointBC = _d_PointBC + blockIdx.x * querySize;
    }
    __syncthreads();

    for (int idx = threadIdx.x + source + 1; idx < nodeNum; idx += blockDim.x) {

      // 1.LCAQueryKernel
      int p = p1;
      int idx2 = rank_[idx + 1];
      int q = euler_tour_pos_[idx2];
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

      int k = 31 - __clz(len);
      int i = 1 << k;

      q = q - i + 1;

      int k_idx = rmq_index_idx[k];
      int RMQIndex_k_p = rmq_index_[k_idx + p], RMQIndex_k_q = rmq_index_[k_idx + q];
      if (tree_height[RMQIndex_k_p] < tree_height[RMQIndex_k_q])
        LCAQuery[idx2] = RMQIndex_k_p;
      else
        LCAQuery[idx2] = RMQIndex_k_q;
    }
    __syncthreads();

    // 2.DistanceCountQueryKernel
    for (int idx = threadIdx.x + source + 1; idx < nodeNum; idx += blockDim.x) {

      int p = source + 1;
      int q = idx + 1;
      int x = x1;
      int ret = 0;
      if (p == q) {
        count[idx] = 1;
        distance[idx] = 0;
      } else {
        int y = rank_[q];      // int r1 = rank[source],
        int lca = LCAQuery[y]; // int LCA = LCAQuery(r1, r2);
        int dis = INT_MAX;
        int cnt = 0;

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

          ret = tree_cnt[tree_cnt_idx[y] + pos]; // count[idx] = tree_cnt[tree_cnt_idx[y] + pos];

          position--;
        }
        int tmp_dis = INT_MAX;
        int dx = tree_dis_idx[x], dy = tree_dis_idx[y];
        int cx = tree_cnt_idx[x], cy = tree_cnt_idx[y];

        for (int i = 0; i <= position; i++) {
          int tmp = tree_dis[dx + i] + tree_dis[dy + i];
          if (tmp_dis > tmp) {
            tmp_dis = tmp;
            cnt = tree_cnt[cx + i] * tree_cnt[cy + i];

          } else if (tmp_dis == tmp) {
            cnt += tree_cnt[cx + i] * tree_cnt[cy + i];
          }
        }

        if (dis > tmp_dis) {
          dis = tmp_dis;
          ret = cnt; // count[idx] = cnt;

        } else if (dis == tmp_dis) {
          ret += cnt; // count[idx] += cnt;
        }

        count[idx] = ret;
        distance[idx] = dis;
      }

      double sigma_t = 1.0 / count[idx];
      int distance_t = distance[idx];
      for (int i = 0; i < querySize; ++i) {
        int i_offset = i * nodeNum;
        if (source != point[i] && idx != point[i] && distance_t == point_distance[i_offset + source] + point_distance[i_offset + idx]) {
          double tmp = point_sigma[i_offset + source] * point_sigma[i_offset + idx] * sigma_t;
          atomicAdd(d_PointBC + i, tmp);
        }
      }
    }
    __syncthreads();
  }
}

void Graph::brandesBCIndexGPU_PointBatch(string roadName, vector<int> &points, const int querySize) {

  int __threadsPerBlock = 256, __blocksPerGrid = (nodenum + threadsPerBlock - 1) / threadsPerBlock;

  // build count index on CPU
  string graph_filename = roadDataPath, index_filename = "tmpRoadCountIndex";
  TreeDecomp td(graph_filename.c_str(), index_filename.c_str());
  td.Reduce();
  td.MakeTree();
  td.MakeIndex();

  // auto t1 = std::chrono::high_resolution_clock::now();

  moveCountIndexToDevice(td);
  initGraphTravel();
  allocateDeviceMemory();
  //   allocateHostMemory();

  int *d_point_distance;
  cudaMalloc(&d_point_distance, querySize * nodenum * sizeof(int));

  int *d_point_sigma;
  cudaMalloc(&d_point_sigma, querySize * nodenum * sizeof(int));

  double *d_PointBC;
  cudaMalloc(&d_PointBC, querySize * blockNum * sizeof(double));
  cudaMemset(d_PointBC, 0, querySize * blockNum * sizeof(double));

  int *d_points;
  cudaMalloc(&d_points, querySize * sizeof(int));
  cudaMemcpy(d_points, points.data(), querySize * sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 0; i < querySize; ++i) {
    int source = points[i];
    LCAQueryKernel<<<__blocksPerGrid, __threadsPerBlock>>>(d_euler_tour_pos_, d_rmq_index_, d_rmq_index_idx, d_tree_height, d_LCAQuery, td.euler_tour_pos_[td.rank_[source + 1]], nodenum);
    // cudaDeviceSynchronize();

    DistanceCountQueryKernel<<<__blocksPerGrid, __threadsPerBlock>>>(d_rank_, d_LCAQuery, d_tree_posSize, d_tree_pos, d_tree_pos_idx, d_tree_dis, d_tree_dis_idx, d_tree_cnt, d_tree_cnt_idx, d_point_distance + nodenum * i, d_point_sigma + nodenum * i, source + 1, td.rank_[source + 1], nodenum);
    cudaDeviceSynchronize();
  }

  // BC_Kernel
  BC_Kernel_merge<<<blocksPerGrid, threadsPerBlock>>>(d_euler_tour_pos_, d_rmq_index_, d_rmq_index_idx, d_tree_height, d_rank_, d_LCAQuery, d_tree_posSize, d_tree_pos, d_tree_pos_idx, d_tree_dis, d_tree_dis_idx, d_tree_cnt, d_tree_cnt_idx,
                                                      d_distance, d_sigma,
                                                      d_edge_vid_vid, d_Neighbor_weight, nodenum, edgenum, blockNum,
                                                      d_point_distance, d_point_sigma, d_PointBC, d_points, querySize);
  cudaDeviceSynchronize();

  h_PointBC.resize(querySize * blockNum);
  cudaMemcpy(h_PointBC.data(), d_PointBC, querySize * blockNum * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < querySize; ++i) {
    for (int j = 1; j < blockNum; ++j) {
      h_PointBC[i] += h_PointBC[j * querySize + i];
    }
  }
  h_PointBC.resize(querySize);

  // auto t2 = std::chrono::high_resolution_clock::now();
  // cout << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << "  seconds (---main kernel---)" << endl;

  freeMemory();
  cudaFree(d_point_distance);
  cudaFree(d_point_sigma);
  cudaFree(d_PointBC);
  cudaFree(d_points);
}

/**

nvcc -std=c++17 -O3 -arch=sm_86  -ccbin /usr/bin/g++-10  RQ.cu  src/TreeCount/TreeDecomp.cpp src/TreeCount/Graph.cpp src/Graph.cpp  src/buildIndex.cpp  -o RQ

./RQ data/roadData/LosAngeles.txt data/queryBatch/LosAngeles.txt 0
./RQ data/roadData/bay.txt data/queryBatch/bay.txt 0

*/

int main(int argc, char *argv[]) {
  cout.precision(8);
  cout << fixed;
  auto t11 = std::chrono::high_resolution_clock::now(), t22 = t11;

  roadDataPath = argv[1];
  queryBatchPath = argv[2];
  int device = stoi(argv[3]);
  cudaSetDevice(device);

  // 1.read graph
  Graph g;
  g.ReadGraph(roadDataPath);
  string roadName = roadDataPath;

  cout << "roadName:" << roadName << "\t" << g.nodenum << "\t" << g.edgenum << "\n";

  // 2.read queried node IDs
  int topk;
  vector<int> randNumber;
  processBactch(g.nodenum, topk, randNumber);
  if (topk >= g.nodenum) {
    topk = g.nodenum;
  }

  // 3.compute BC values
  int querySize = topk;
  t11 = std::chrono::high_resolution_clock::now();
  g.brandesBCIndexGPU_PointBatch(roadName, randNumber, querySize);
  t22 = std::chrono::high_resolution_clock::now();

  double bcTime = std::chrono::duration_cast<std::chrono::duration<double>>(t22 - t11).count();
  cout << bcTime << "  seconds (---totalTime---)" << endl;

  // 4.print BC values
  cout << left;
  cout << endl;
  cout << setw(10) << "vertexID" << setw(10) << "BCValue" << endl;
  cout << string(20, '-') << endl;

  for (int i = 0; i < topk; ++i) {
    cout << setw(10) << (randNumber[i] + 1) << setw(10)
         << setw(10) << g.h_PointBC[i] << endl;
  }
  cout << endl;
}

void processBactch(int nodenum, int &querySize, vector<int> &randNumber) {
  ifstream ifs;
  ifs.open(queryBatchPath, ios::in);
  if (!ifs) {
    cout << "Cannot open: " << queryBatchPath << endl;
  }
  ifs >> querySize;

  int idd;
  for (int i = 0; i < querySize; ++i) {
    ifs >> idd;
    --idd;
    randNumber.push_back(idd);
  }

  if (randNumber.size() != querySize) {
    cout << "randNumber.size()<" << randNumber.size() << "  querySize=" << querySize << endl;
    exit(1);
  }

  cout << "The number of vertices to query: " << querySize << endl;
}