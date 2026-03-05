#pragma once
#define _CRT_SECURE_NO_DEPRECATE

#include <assert.h>

#include <cstdio>
#include <map>
#include <utility>
#include <vector>

using std::map;
using std::vector;

class Graph_
{
public:
    int n_, m_;               // number of nodes & edges
    vector<map<int, int>> E_; // record edges and edge weight: u--->(v, dis)


    vector<map<int, int>> my_E_; // hop

    vector<int> D_;           // degree

public:
    Graph_();
    Graph_(const char *file);
    void ReadGraph(const char *file);
    void ReadWeightedGraph(const char *file);
    bool isEdgeExist(int u, int v);
    void insertEdge(int u, int v, int w);
    void deleteEdge(int u, int v);

    inline int n() const { return n_; }
    inline int m() const { return m_; }
};
