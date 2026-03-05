#pragma once
#include "../src/TreeCount/TreeDecomp.h"
#include "std.h"
using namespace std;
struct Edge {
  int to;
  int length;
  Edge(int t, int l) : to(t), length(l) {}
};
struct Nei {
  int nid;
  int w;
  int c;
};

struct Node {
  Node() {
    vert.clear();
    ch.clear();
    pos.clear();
    dis.clear();
    cnt.clear();
    FN.clear();
    DisRe.clear();
    pa = -1;
    uniqueVertex = -1;
    height = 0;
    hdepth = 0;
    // neighInf.clear();
    // changedPos.clear();
  }
  // tree node
  vector<pair<int, pair<int, int>>> vert; // neighID/weight/count
  vector<int> ch;                         //
  vector<int> pos;                        //
  vector<int> dis, cnt;                   // the distance value and corresponding count number
  vector<bool> FN;                        // another succint way of FromNode
  set<int> DisRe;                         // record the star weight change (causing the distance change)
  int pa;                                 // parent
  int uniqueVertex;
  int height, hdepth; // hdepty is the deepest node that a vertex still exists//height from 1

  // for Decrease2HopBatch5
  vector<int> VidtoTNid;
  vector<pair<pair<int, int>, int>> disCheck; // uh ah disBF
  vector<pair<int, int>> disChange;           // dis(u,a) change (rank) to dis

  bool isBranchNode = false;
  bool wbranchNode = false;

  bool branchNodePair = false;
  bool highFrequencyNode = false;
  //  OptimizeVS
  vector<int> posOptimize;
  vector<int> vertOptimize;

  //  ProjectVS
  vector<vector<int>> posProject_3;

  // P2H
  vector<vector<int>> posP2H_3;
};

class Graph {
public:
  string buildIndexMethod;
  int (Graph::*myQuery)(int, int) = nullptr;

public:
  vector<vector<Edge>> nbrs;
  vector<double> BC;
  vector<bool> isDijSearch;
  vector<vector<int>> successors;
  vector<vector<int>> predecessors;
  vector<long int> sigma;
  vector<int> distance;
  vector<bool> closed;
  vector<int> st;
  vector<double> delta; // for  backward step

public:
  /*
   * read graph
   */
  int nodenum;
  int edgenum;
  vector<pair<int, int>> Edges;            //(ID1,ID2) the order is the edge ID
  vector<vector<pair<int, int>>> Neighbor; // Neighbor[ID1].push_back(make_pair(ID2, weight));
  vector<map<int, pair<int, int>>> E;
  void ReadGraph(string graphname);
  int DijkstraDis(int ID1, int ID2);

  /*
   * contraction with order file
   */
  vector<int> NodeOrder;  // NodeOrder[nodeID] = nodeorder;
  vector<int> vNodeOrder; // its inverted list :vNodeOrder[nodeorder] = nodeID;
  // vector<int> order;									   // node ID is added in increasing order  //vNodeOrder
  vector<vector<pair<int, pair<int, int>>>> NeighborCon; // tree node(ID2,(weight,1))
  // vector<map<int, vector<int>>> SCconNodesMT;			   // Graph::CHconsorderMT
  map<pair<int, int>, vector<int>> SCconNodes; // a--b shortcut: potential contract nodes
  void CHconsorderMT(string orderfile);        // easier one
  void deleteEorder(int u, int v);
  void insertEorder(int u, int v, int w);

  vector<int> DD, DD2;
  void GraphConstract(string orderfileWritePath, string beizhu);
  void insertE(int u, int v, int w);
  void deleteE(int u, int v);

  /*
   * make tree
   */
  int heightMax;
  vector<vector<int>> VidtoTNid; // one vertex exist in those tree nodes (nodeID--->tree node rank)
  vector<int> rank;              // rank[vertexid]
  vector<Node> Tree;
  void makeTree();
  int match(int x, vector<pair<int, pair<int, int>>> &vert);

  /*
   * make index
   */
  void makeIndex();
  void makeIndexDFS(int p, vector<int> &list);

  /*
   * prepare for the LCA calculation
   */
  vector<int> EulerSeq;
  vector<int> toRMQ;
  vector<vector<int>> RMQIndex;
  void makeRMQ();
  void makeRMQDFS(int p, int height);

  /*
   * LCA query
   */
  int LCAQuery(int _p, int _q);

  /*
   * optimizeVS
   */
  bool isOptimizeVS = false;
  void optimizeVS();

  /*
   * projectVS
   */
  void projectVS_3(int vID);
  int descendentNumberDFS(vector<int> &descendentNumber, int r);
  void SVSP_ProjectVS(int k, int d, int R, vector<int> &res);

  /*
   *  P2H: optimizeVS+projectVS
   */
  void SVSP_P2H(int k, int d, int R, vector<int> &res);
  void P2H_3(vector<int> &vID);

public:
  /**
   *GPU BC
   */
  void brandesBCIndexGPU_5(string roadName); // dis + count

  int *d_toRMQ = nullptr, *d_RMQIndex = nullptr, *d_RMQIndex_idx = nullptr, *d_rank = nullptr;
  int *d_Tree_height = nullptr, *d_Tree_posOptimize = nullptr, *d_Tree_posOptimize_idx = nullptr, *d_Tree_dis = nullptr, *d_Tree_dis_idx = nullptr;
  bool *d_Tree_branch = nullptr;
  int *d_Tree_posP2H_3 = nullptr, *d_Tree_posP2H_3_idx1 = nullptr, *d_Tree_posP2H_3_idx2 = nullptr;
  void moveDisIndexToDevice();

  int *d_rank_ = nullptr, *d_euler_tour_pos_ = nullptr, *d_rmq_index_ = nullptr, *d_rmq_index_idx = nullptr;
  int *d_tree_height = nullptr, *d_tree_posSize = nullptr, *d_tree_pos = nullptr, *d_tree_pos_idx = nullptr, *d_tree_dis = nullptr, *d_tree_dis_idx = nullptr, *d_tree_cnt = nullptr, *d_tree_cnt_idx = nullptr, *d_my_tree_cnt = nullptr, *d_my_tree_cnt_idx = nullptr;
  void moveCountIndexToDevice(TreeDecomp &td);

  vector<int> h_Neighbor_NodeID_flat, h_Neighbor_weight_flat, h_Neighbor_idx, h_edge_vid_vid;
  int *d_Neighbor_NodeID = nullptr, *d_Neighbor_weight = nullptr, *d_Neighbor_idx = nullptr, *d_edge_vid_vid = nullptr;
  void initGraphTravel();

  unsigned char *d_successors_char = nullptr;
  int *d_indegree = nullptr, *d_LCA = nullptr, *d_LCAQuery = nullptr, *d_distance = nullptr, *d_sigma = nullptr;
  double *d_delta = nullptr, *d_BC = nullptr;
  double *d_delta_double = nullptr, *tmp_BC_double = nullptr, *d_BC_double = nullptr;
  int *d_queue = nullptr, *d_queue_edge = nullptr, *d_cnt = nullptr, *d_cnt_edge = nullptr, *d_layerSize = nullptr, *d_layerQueueStart = nullptr;
  int *d_layer = nullptr, *d_layer_edge = nullptr, *d_layer_size = nullptr, *d_layer_start = nullptr, *d_maxLayer = nullptr;

  unsigned char *h_successors_char = nullptr;
  int *h_indegree = nullptr, *h_sigma = nullptr;
  double *h_delta = nullptr, *h_BC = nullptr;
  // int *h_queue = nullptr, *h_queue_edge = nullptr;
  int *h_layer = nullptr, *h_layer_edge = nullptr, *h_layer_size = nullptr, *h_layer_start = nullptr;

  void allocateDeviceMemory();
  void allocateHostMemory();
  void freeMemory();

  // int source;
  vector<double> h_PointBC;
  void brandesBCIndexGPU_PointBatch(string roadName, vector<int> &points, const int querySize);
};

namespace benchmark {

#define NULLINDEX 0xFFFFFFFF

template <int log_k, typename k_t, typename id_t>
class heap {

public:
  // Expose types.
  typedef k_t key_t;
  typedef id_t node_t;

  // Some constants regarding the elements.
  // static const node_t NULLINDEX = 0xFFFFFFFF;
  static const node_t k = 1 << log_k;

  // A struct defining a heap element.
  struct element_t {
    key_t key;
    node_t element;

    element_t() : key(0), element(0) {}

    element_t(const key_t k, const node_t e) : key(k), element(e) {}
  };

public:
  // Constructor of the heap.
  explicit heap(node_t n) : n(0), max_n(n), elements(n), position(n, NULLINDEX) {
  }

  heap() = default;

  // Size of the heap.
  inline node_t size() const {
    return n;
  }

  // Heap empty?
  inline bool empty() const {
    return size() == 0;
  }

  // Extract min element.
  inline void extract_min(node_t &element, key_t &key) {
    assert(!empty());

    element_t &front = elements[0];

    // Assign element and key.
    element = front.element;
    key = front.key;

    // Replace elements[0] by last element.
    position[element] = NULLINDEX;
    --n;
    if (!empty()) {
      front = elements[n];
      position[front.element] = 0;
      sift_down(0);
    }
  }

  inline key_t top() {
    assert(!empty());

    element_t &front = elements[0];

    return front.key;
  }

  inline node_t top_value() {

    assert(!empty());

    element_t &front = elements[0];

    return front.element;
  }

  // Update an element of the heap.
  inline void update(const node_t element, const key_t key) {
    if (position[element] == NULLINDEX) {
      element_t &back = elements[n];
      back.key = key;
      back.element = element;
      position[element] = n;
      sift_up(n++);
    } else {
      node_t el_pos = position[element];
      element_t &el = elements[el_pos];
      if (key > el.key) {
        el.key = key;
        sift_down(el_pos);
      } else {
        el.key = key;
        sift_up(el_pos);
      }
    }
  }

  // Clear the heap.
  inline void clear() {
    for (node_t i = 0; i < n; ++i) {
      position[elements[i].element] = NULLINDEX;
    }
    n = 0;
  }

  // Cheaper clear.
  inline void clear(node_t v) {
    position[v] = NULLINDEX;
  }

  inline void clear_n() {
    n = 0;
  }

  // Test whether an element is contained in the heap.
  inline bool contains(const node_t element) const {
    return position[element] != NULLINDEX;
  }

protected:
  // Sift up an element.
  inline void sift_up(node_t i) {
    assert(i < n);
    node_t cur_i = i;
    while (cur_i > 0) {
      node_t parent_i = (cur_i - 1) >> log_k;
      if (elements[parent_i].key > elements[cur_i].key)
        swap(cur_i, parent_i);
      else
        break;
      cur_i = parent_i;
    }
  }

  // Sift down an element.
  inline void sift_down(node_t i) {
    assert(i < n);

    while (true) {
      node_t min_ind = i;
      key_t min_key = elements[i].key;

      node_t child_ind_l = (i << log_k) + 1;
      node_t child_ind_u = std::min(child_ind_l + k, n);

      for (node_t j = child_ind_l; j < child_ind_u; ++j) {
        if (elements[j].key < min_key) {
          min_ind = j;
          min_key = elements[j].key;
        }
      }

      // Exchange?
      if (min_ind != i) {
        swap(i, min_ind);
        i = min_ind;
      } else {
        break;
      }
    }
  }

  // Swap two elements in the heap.
  inline void swap(const node_t i, const node_t j) {
    element_t &el_i = elements[i];
    element_t &el_j = elements[j];

    // Exchange positions
    position[el_i.element] = j;
    position[el_j.element] = i;

    // Exchange elements
    element_t temp = el_i;
    el_i = el_j;
    el_j = temp;
  }

private:
  // Number of elements in the heap.
  node_t n;

  // Number of maximal elements.
  node_t max_n;

  // Array of length heap_elements.
  vector<element_t> elements;

  // An array of positions for all elements.
  vector<node_t> position;
};
} // namespace benchmark
