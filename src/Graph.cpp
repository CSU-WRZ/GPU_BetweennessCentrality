#include "../include/Graph.h"
// #define PRUNE_EDGE

void Graph::ReadGraph(string graphname)
{
    ifstream IF(graphname);
    if (!IF) {
        cout << "Cannot open Map " << graphname << endl;
    }

    IF >> nodenum >> edgenum;
    // cout << "nodenum=" << nodenum << " edgenum=" << edgenum << endl;
    vector<pair<int, int>> vecp;
    Neighbor.assign(nodenum, vecp);
    map<int, pair<int, int>> m;
    E.assign(nodenum, m);

    // to avoid the redundant information
    set<pair<int, int>> EdgeRedun;

    int ID1, ID2, weight;
    for (int i = 0; i < edgenum; i++) {
        IF >> ID1 >> ID2 >> weight;
        ID1 -= 1;
        ID2 -= 1;

        if (weight == 0)
            weight = 1;
        if (EdgeRedun.find(make_pair(ID1, ID2)) == EdgeRedun.end()) {
            Neighbor[ID1].push_back(make_pair(ID2, weight));
            E[ID1].insert(make_pair(ID2, make_pair(weight, 1)));
        } else {
            cout << "redun  " << ID1 << " " << ID2 << " " << weight << endl;
        }
        EdgeRedun.insert(make_pair(ID1, ID2));
    }

    IF.close();
#ifdef PRUNE_EDGE
    vector<map<int, pair<int, int>>> Neig;
    Neig.resize(nodenum);
    for (int i = 0; i < nodenum; ++i) {
        for (int j = 0; j < Neighbor[i].size(); ++j) {
            Neig[i][Neighbor[i][j].first] = make_pair(Neighbor[i][j].second, 1);
        }
    }
    int cntPruneEdge = 0;

    for (int v = 0; v < nodenum; ++v) {
        for (int i = 0; i < Neighbor[v].size(); ++i) {
            int neigID1 = Neighbor[v][i].first;
            int neigWeight1 = Neighbor[v][i].second;

            for (int j = i + 1; j < Neighbor[v].size(); ++j) {
                int neigID2 = Neighbor[v][j].first;
                int neigWeight2 = Neighbor[v][j].second;

                if (Neig[neigID1].find(neigID2) != Neig[neigID1].end()) {
                    if (Neig[neigID1][neigID2].first >= (neigWeight1 + neigWeight2)) {
                        Neig[neigID1][neigID2].second = -1;
                        Neig[neigID2][neigID1].second = -1;
                        cntPruneEdge++;
                    }
                }
            }
        }
    }
    cout << "prune edge is " << cntPruneEdge << endl;
    for (ID1 = 0; ID1 < nodenum; ++ID1) {
        Neighbor[ID1].clear();
        E[ID1].clear();

        for (auto &p : Neig[ID1]) {
            if (p.second.second != -1) {
                ID2 = p.first;
                weight = p.second.first;
                Neighbor[ID1].push_back(make_pair(ID2, weight));
                E[ID1].insert(make_pair(ID2, make_pair(weight, 1)));
            }
        }
    }

#endif
    // cout << "finish read graph" << endl;
}

