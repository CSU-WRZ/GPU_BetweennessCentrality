
#include "../include/Graph.h"

void Graph::CHconsorderMT(string orderfile)
{
    ifstream IF(orderfile);
    if (!IF) {
        cout << "Cannot open Map " << orderfile << endl;
    }
    NodeOrder.assign(nodenum, -1);
    vNodeOrder.assign(nodenum, -1);
    int num, nodeID, nodeorder;
    IF >> num;
    for (int i = 0; i < num; i++) {
        IF >> nodeID >> nodeorder;
        NodeOrder[nodeID] = nodeorder;
        if (nodeorder != -1) {
            vNodeOrder[nodeorder] = nodeID;
        }
    }
    IF.close();

    vector<pair<int, pair<int, int>>> vect;
    NeighborCon.assign(nodenum, vect);

    vector<bool> exist;
    exist.assign(nodenum, true);

    for (int nodeorder = 0; nodeorder < nodenum; nodeorder++) {
        int x = vNodeOrder[nodeorder];
        if (x != -1) { // to identify and exclude the isolated vertices
            exist[x] = false;

            vector<pair<int, pair<int, int>>> Neigh;

            for (auto it = E[x].begin(); it != E[x].end(); it++) {
                if (exist[(*it).first]) {
                    Neigh.push_back(*it); // （ID2, (weight,1)）
                }
            }

            for (int i = 0; i < Neigh.size(); i++) {
                int y = Neigh[i].first;
                deleteEorder(x, y);
            }

            NeighborCon[x].assign(Neigh.begin(), Neigh.end());

            if (Neigh.size() <= 1000) {

                // single thread
                // shortcut E
                for (int i = 0; i < Neigh.size(); i++) {
                    for (int j = i + 1; j < Neigh.size(); j++) {
                        insertEorder(Neigh[i].first, Neigh[j].first, Neigh[i].second.first + Neigh[j].second.first);
                        if (Neigh[i].first < Neigh[j].first) {
                            SCconNodes[make_pair(Neigh[i].first, Neigh[j].first)].push_back(x);
                        } else if (Neigh[j].first < Neigh[i].first) {
                            SCconNodes[make_pair(Neigh[j].first, Neigh[i].first)].push_back(x);
                        }
                    }
                }
            } else {
                cout << "Graph::CHconsorderMT\t\t:" << Neigh.size() << " tree width >1000 and use multiple thread " << endl;

#ifdef PARAEELE
                // // multiple thread
                cout << "Graph::CHconsorderMT\t\t:" << Neigh.size() << " tree width >1000 and use multiple thread " << endl;
#pragma omp parallel for num_threads(NUM_THREAD)
                for (int i = 0; i < Neigh.size(); i++) {
                    for (int j = 0; j < Neigh.size(); j++) {
                        if (i == j) {
                            continue;
                        }
                        int u = Neigh[i].first;
                        int v = Neigh[j].first;
                        int w = Neigh[i].second.first + Neigh[j].second.first;

                        if (E[u].find(v) == E[u].end()) {
                            E[u].insert(make_pair(v, make_pair(w, 1)));
                        } else {
                            if (E[u][v].first > w) {
                                E[u][v] = make_pair(w, 1);
                            } else if (E[u][v].first == w) {
                                E[u][v].second += 1;
                            }
                        }
                        if (Neigh[i].first < Neigh[j].first) {
                            SCconNodes[make_pair(Neigh[i].first, Neigh[j].first)].push_back(x);
                        }
                    }
                }

                // int step = Neigh.size() / threadnum;
                // boost::thread_group thread;
                // for (int i = 0; i < threadnum; i++)
                // {
                // 	pair<int, int> p;
                // 	p.first = i * step;
                // 	if (i == threadnum - 1)
                // 		p.second = Neigh.size();
                // 	else
                // 		p.second = (i + 1) * step;
                // 	thread.add_thread(new boost::thread(&Graph::NeighborComorder, this, boost::ref(Neigh), p, x));
                // }
                // thread.join_all();
#endif
            }
        }
    }
}

void Graph::deleteEorder(int u, int v)
{
    if (E[u].find(v) != E[u].end()) {
        E[u].erase(E[u].find(v));
        // DD[u]--;
    }

    if (E[v].find(u) != E[v].end()) {
        E[v].erase(E[v].find(u));
        // DD[v]--;
    }
}
void Graph::insertEorder(int u, int v, int w)
{

    if (E[u].find(v) == E[u].end()) {
        E[u].insert(make_pair(v, make_pair(w, 1)));
        // DD[u]++;
        // DD2[u]++;
    } else {
        if (E[u][v].first > w)
            E[u][v] = make_pair(w, 1);
        else if (E[u][v].first == w)
            E[u][v].second += 1;
    }

    if (E[v].find(u) == E[v].end()) {
        E[v].insert(make_pair(u, make_pair(w, 1)));
        // DD[v]++;
        // DD2[v]++;
    } else {
        if (E[v][u].first > w)
            E[v][u] = make_pair(w, 1);
        else if (E[v][u].first == w)
            E[v][u].second += 1;
    }
}

void Graph::makeTree()
{
    vector<int> vecemp;
    VidtoTNid.assign(nodenum, vecemp);

    rank.assign(nodenum, 0);
    // Tree.clear();
    int len = vNodeOrder.size() - 1;
    heightMax = 0;

    Node rootn;
    int x = vNodeOrder[len];
    rootn.vert = NeighborCon[x];
    rootn.uniqueVertex = x;
    rootn.pa = -1;
    rootn.height = 1;
    rank[x] = 0;
    Tree.push_back(rootn);
    len--;

    int nn;
    for (; len >= 0; len--) {
        int x = vNodeOrder[len];
        Node nod;
        nod.vert = NeighborCon[x];

        nod.uniqueVertex = x;
        int pa = match(x, NeighborCon[x]);
        Tree[pa].ch.push_back(Tree.size());
        nod.pa = pa;
        nod.height = Tree[pa].height + 1;

        nod.hdepth = Tree[pa].height + 1;
        for (int i = 0; i < NeighborCon[x].size(); i++) {
            nn = NeighborCon[x][i].first;
            VidtoTNid[nn].push_back(Tree.size());
            if (Tree[rank[nn]].hdepth < Tree[pa].height + 1)
                Tree[rank[nn]].hdepth = Tree[pa].height + 1;
        }

        if (nod.height > heightMax)
            heightMax = nod.height;
        rank[x] = Tree.size();
        Tree.push_back(nod);
    }
}
int Graph::match(int x, vector<pair<int, pair<int, int>>> &vert)
{
    int nearest = vert[0].first;
    // if(vert.size()>1){
    for (int i = 1; i < vert.size(); i++) {
        if (rank[vert[i].first] > rank[nearest])
            nearest = vert[i].first;
    }
    int p = rank[nearest];
    return p;
}
void Graph::makeIndex()
{
    makeRMQ();

    // initialize
    vector<int> list; // list.clear();
    list.push_back(Tree[0].uniqueVertex);
    Tree[0].pos.clear();
    Tree[0].pos.push_back(0);

    for (int i = 0; i < Tree[0].ch.size(); i++) {
        makeIndexDFS(Tree[0].ch[i], list);
    }
}
void Graph::makeIndexDFS(int p, vector<int> &list)
{
    // initialize
    int NeiNum = Tree[p].vert.size();
    Tree[p].pos.assign(NeiNum + 1, 0);
    Tree[p].dis.assign(list.size(), INF);
    Tree[p].cnt.assign(list.size(), 0);
    Tree[p].FN.assign(list.size(), true);

    // pos
    // map<int,Nei> Nmap; Nmap.clear();//shortcut infor ordered by the pos ID
    for (int i = 0; i < NeiNum; i++) {
        for (int j = 0; j < list.size(); j++) {
            if (Tree[p].vert[i].first == list[j]) {
                Tree[p].pos[i] = j;
                Tree[p].dis[j] = Tree[p].vert[i].second.first;
                Tree[p].cnt[j] = 1;
                break;
            }
        }
    }
    Tree[p].pos[NeiNum] = list.size();

    // dis
    for (int i = 0; i < NeiNum; i++) {
        int x = Tree[p].vert[i].first;
        int disvb = Tree[p].vert[i].second.first;
        int k = Tree[p].pos[i]; // the kth ancestor is x

        for (int j = 0; j < list.size(); j++) {
            int y = list[j]; // the jth ancestor is y

            int z; // the distance from x to y
            if (k != j) {
                if (k < j)
                    z = Tree[rank[y]].dis[k];
                else if (k > j)
                    z = Tree[rank[x]].dis[j];

                if (Tree[p].dis[j] > z + disvb) {
                    Tree[p].dis[j] = z + disvb;
                    Tree[p].FN[j] = false;
                    Tree[p].cnt[j] = 1;
                } else if (Tree[p].dis[j] == z + disvb) {
                    Tree[p].cnt[j] += 1;
                }
            }
        }
    }

    // nested loop
    list.push_back(Tree[p].uniqueVertex);
    for (int i = 0; i < Tree[p].ch.size(); i++) {
        makeIndexDFS(Tree[p].ch[i], list);
    }
    list.pop_back();
}
void Graph::makeRMQ()
{
    // EulerSeq.clear();
    toRMQ.assign(nodenum, 0);
    // RMQIndex.clear();
    makeRMQDFS(0, 1);
    RMQIndex.push_back(EulerSeq);

    int m = EulerSeq.size();
    for (int i = 2, k = 1; i < m; i = i * 2, k++) {
        vector<int> tmp;
        // tmp.clear();
        tmp.assign(m, 0);
        for (int j = 0; j < m - i; j++) {
            int x = RMQIndex[k - 1][j], y = RMQIndex[k - 1][j + i / 2];
            if (Tree[x].height < Tree[y].height)
                tmp[j] = x;
            else
                tmp[j] = y;
        }
        RMQIndex.push_back(tmp);
    }
}
void Graph::makeRMQDFS(int p, int height)
{
    toRMQ[p] = EulerSeq.size();
    EulerSeq.push_back(p);
    for (int i = 0; i < Tree[p].ch.size(); i++) {
        makeRMQDFS(Tree[p].ch[i], height + 1);
        EulerSeq.push_back(p);
    }
}

// optimize
void Graph::optimizeVS()
{

    for (int v = 0; v < nodenum; ++v) {
        int childNumber = Tree[v].ch.size();
        int treeVPosBack = Tree[v].pos.back();
        int uniqueVertex = Tree[v].uniqueVertex;
        if (childNumber < 2) {
            Tree[v].posOptimize.push_back(treeVPosBack);
            Tree[v].vertOptimize.push_back(uniqueVertex);
        } else if (childNumber == 2) {
            int child1 = Tree[v].ch[0], child2 = Tree[v].ch[1];
            int tmp = Tree[child1].pos.size() < Tree[child2].pos.size() ? child1 : child2;
            for (int k = Tree[tmp].pos.size() - 2; k >= 0; --k) {
                if (Tree[tmp].pos[k] != treeVPosBack) {
                    Tree[v].posOptimize.push_back(Tree[tmp].pos[k]);
                    Tree[v].vertOptimize.push_back(Tree[tmp].vert[k].first);
                }
            }

            Tree[v].posOptimize.push_back(treeVPosBack);
            Tree[v].vertOptimize.push_back(uniqueVertex);
        } else {

            unordered_map<int, int> count;
            unordered_map<int, int> vert;

            for (int i = Tree[v].pos.size() - 1; i >= 0; --i) {
                vert[Tree[v].pos[i]] = Tree[v].vert[i].first;
            }

            for (int i = 0; i < childNumber; ++i) {
                int _ch = Tree[v].ch[i];
                for (int k = Tree[_ch].pos.size() - 2; k >= 0; --k) {
                    int u = Tree[_ch].pos[k];
                    count[u] += 1;
                }
            }

            vector<int> A(childNumber, 0);
            for (int i = 0; i < childNumber; ++i) {
                int _ch = Tree[v].ch[i];
                for (int k = Tree[_ch].pos.size() - 2; k >= 0; --k) {
                    int u = Tree[_ch].pos[k];
                    if (count[u] == 1) {
                        A[i] += 1;
                    }
                }
            }
            int idx = 0, maxA = A[idx];
            for (int i = 1; i < childNumber; ++i) {
                if (A[i] > maxA) {
                    idx = i;
                    maxA = A[i];
                }
            }
            for (int _ch = Tree[v].ch[idx], k = Tree[_ch].pos.size() - 2; k >= 0; --k) {
                int u = Tree[_ch].pos[k];
                count[u] -= 1;
            }

            for (auto &[key, value] : count) {

                if ((value > 0) && (key != treeVPosBack)) {
                    Tree[v].posOptimize.push_back(key);
                    Tree[v].vertOptimize.push_back(vert[key]);
                }
            }
            Tree[v].posOptimize.push_back(treeVPosBack);
            Tree[v].vertOptimize.push_back(uniqueVertex);
        }
    }
    isOptimizeVS = true;
}

int Graph::descendentNumberDFS(vector<int> &descendentNumber, int r)
{
    for (int i = 0; i < Tree[r].ch.size(); i++) {
        int n = descendentNumberDFS(descendentNumber, Tree[r].ch[i]);
        descendentNumber[r] += (n + 1);
    }
    return descendentNumber[r];
}

void Graph::SVSP_ProjectVS(int k, int d, int R, vector<int> &res)
{
    cout << "k=" << k << " d=" << d << " R=" << R << endl;

    /**
     * @brief
     * find top k brance nodes with higest p
     */
    // t1 = std::chrono::high_resolution_clock::now();
    vector<int> Descendent(nodenum, 0);
    descendentNumberDFS(Descendent, 0);
    vector<pair<double, int>> s1; //  pair(p[v],nodeid>
    for (int id = 0; id < nodenum; ++id) {
        int rankID = rank[id], childNumber = Tree[rankID].ch.size();
        if (childNumber < 2) {
            continue;
        } else {
            double p = 0;
            for (int i = 0; i < childNumber; ++i) {
                for (int j = i + 1; j < childNumber; ++j)
                    p += 1.0 * (Descendent[Tree[rankID].ch[i]] + 1) * (Descendent[Tree[rankID].ch[j]] + 1);
            }
            p = 2 * p / nodenum / nodenum;
            s1.push_back({p, id});
        }
    }
    sort(s1.begin(), s1.end(), [](const pair<double, int> &l, const pair<double, int> &r) { return l.first > r.first; });

    /**
     * @brief
     * project k nodes
     */
    unordered_map<int, double> BN;  // top k branchnode , nodeid -> p[v]
    unordered_map<int, double> AHS; // nodeid->0
    for (int i = 0; i < k; ++i) {
        int id = s1[i].second;
        double p = s1[i].first;
        BN[id] = p;
        AHS[id] = 0;
    }

    for (auto &bn : BN) {
        projectVS_3(bn.first);
    }

    /**
     * @brief
     * generate a set RQ of R random queries
     * estimate ΔAHS based on BN and RQ
     */
    srand(time(NULL));
    for (int i = 0; i < R; ++i) {
        int ID1 = rand() % nodenum, ID2 = rand() % nodenum;
        if (ID1 == ID2) {
            i--;
            continue;
        }
        int r1 = rank[ID1], r2 = rank[ID2];
        int LCA = LCAQuery(r1, r2);
        if ((LCA == r1) || (LCA == r2)) {
            continue;
        }
        if (BN.find(Tree[LCA].uniqueVertex) != BN.end()) {
            int minID = Tree[LCA].posProject_3[ID1].size() < Tree[LCA].posProject_3[ID2].size() ? ID1 : ID2;
            AHS[Tree[LCA].uniqueVertex] += (Tree[LCA].pos.size() - Tree[LCA].posProject_3[minID].size());
        }
    }

    vector<pair<double, int>> s2; //  pair(ΔAHS,nodeid）
    for (auto &[id, _] : AHS)
        s2.push_back({AHS[id], id});

    sort(s2.begin(), s2.end(), [](const pair<double, int> &l, const pair<double, int> &r) { return l.first > r.first; });

    cout << "\tthe top " << d << " probability vertex ID is :" << endl;
    for (int i = 0; i < d; ++i) {
        int id = s2[i].second;
        double p = s2[i].first;
        cout << "\t  " << id << "\t" << p << endl;
    }
    cout << endl;

    /**
     * @brief
     * remove Z(v,s) for all descendents N(s) of N(v) for N(v) ∉ A  (res)
     * bool isbranceNode
     */
    unordered_set<int> tmpRes; // top d branchnode ranked by ΔAHS
    for (int i = 0; i < d; ++i)
        tmpRes.insert(s2[i].second);

    for (auto &p : BN) {
        if (tmpRes.find(p.first) == tmpRes.end()) {
            Tree[rank[p.first]].isBranchNode = false;
            Tree[rank[p.first]].posProject_3.clear();
        }
    }
    for (auto &p : tmpRes)
        res.push_back(p);
}

// project
void Graph::projectVS_3(int vID)
{
    int v = rank[vID];
    if (Tree[v].ch.size() < 2) {
        cout << "vID=" << vID << "  Tree[v].ch.size: " << Tree[v].ch.size() << endl;
        return;
    }
    if (Tree[v].posProject_3.size() == 0) {
        Tree[v].posProject_3.resize(nodenum);
    }

    Tree[v].isBranchNode = true;

    vector<int> childID;
    childID.reserve(nodenum);

    queue<int> q;
    q.push(v);
    while (!q.empty()) {
        int top = q.front();
        q.pop();
        for (auto &p : Tree[top].ch) {
            q.push(p);
            childID.push_back(Tree[p].uniqueVertex);
        }
    }

    for (auto &s : childID) {
        vector<int> &posProject = Tree[v].posProject_3[s];
        int posSize = Tree[v].pos.size();
        s = rank[s];
        for (int c = 0; c < posSize; ++c) {
            bool flag = true;
            int d_s_c = Tree[s].dis[Tree[v].pos[c]];

            for (int u = 0; u < posSize; ++u) {
                if (u == c) {
                    continue;
                }
                int d_s_u = Tree[s].dis[Tree[v].pos[u]];
                if (d_s_u > d_s_c)
                    continue;
                // dis(u,c)
                int d_u_c = -1;
                int low = Tree[v].pos[u] > Tree[v].pos[c] ? u : c;
                int high = u + c - low;

                if (low == (posSize - 1)) {
                    d_u_c = Tree[v].dis[Tree[v].pos[high]];
                } else {
                    d_u_c = Tree[rank[Tree[v].vert[low].first]].dis[Tree[v].pos[high]];
                }
                if (d_s_c == (d_s_u + d_u_c)) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                posProject.push_back(Tree[v].pos[c]);
            }
        }

        // Tree[v].posProject_3[s] = posProject;
    }
}

void Graph::SVSP_P2H(int k, int d, int R, vector<int> &res)
{
    // cout << "k=" << k << " d=" << d << " R=" << R << endl;
    /**
     * find top k brance nodes with higest p
     */
    vector<int> Descendent(nodenum, 0);
    descendentNumberDFS(Descendent, 0);
    vector<pair<double, int>> s1; // pair(p[v],nodeid>
    for (int id = 0; id < nodenum; ++id) {
        int rankID = rank[id], childNumber = Tree[rankID].ch.size();
        if (childNumber < 2) {
            continue;
        } else {
            double p = 0;
            for (int i = 0; i < childNumber; ++i) {
                for (int j = i + 1; j < childNumber; ++j)
                    p += 1.0 * (Descendent[Tree[rankID].ch[i]] + 1) * (Descendent[Tree[rankID].ch[j]] + 1);
            }
            p = 2 * p / nodenum / nodenum;
            s1.push_back({p, id});
        }
    }
    sort(s1.begin(), s1.end(), [](const pair<double, int> &l, const pair<double, int> &r) { return l.first > r.first; });

    /**
     * project k nodes
     */
    unordered_map<int, double> BN;  // top k branchnode , nodeid -> p[v]
    unordered_map<int, double> AHS; // nodeid->0
    vector<int> vIDSet;
    for (int i = 0; i < k; ++i) {
        int id = s1[i].second;
        double p = s1[i].first;
        BN[id] = p;
        AHS[id] = 0;
        vIDSet.push_back(id);
    }
    P2H_3(vIDSet);
    /**
     * generate a set RQ of R random queries
     * estimate ΔAHS based on BN and RQ
     */
    srand(time(NULL));
    for (int i = 0; i < R; ++i) {
        int ID1 = rand() % nodenum, ID2 = rand() % nodenum;
        if (ID1 == ID2) {
            i--;
            continue;
        }
        int r1 = rank[ID1], r2 = rank[ID2];
        int LCA = LCAQuery(r1, r2);
        if ((LCA == r1) || (LCA == r2)) {
            continue;
        }
        if (BN.find(Tree[LCA].uniqueVertex) != BN.end()) {
            int minID = Tree[LCA].posP2H_3[ID1].size() < Tree[LCA].posP2H_3[ID2].size() ? ID1 : ID2;
            AHS[Tree[LCA].uniqueVertex] += (Tree[LCA].posOptimize.size() - Tree[LCA].posP2H_3[minID].size());
        }
    }

    vector<pair<double, int>> s2; //  pair(ΔAHS,nodeid）
    for (auto &[id, _] : AHS)
        s2.push_back({AHS[id], id});

    sort(s2.begin(), s2.end(), [](const pair<double, int> &l, const pair<double, int> &r) { return l.first > r.first; });

    /**
     * remove Z(v,s) for all descendents N(s) of N(v) for N(v) ∉ A  (res)
     * bool isbranceNode
     */
    unordered_set<int> tmpRes; // top d branchnode ranked by ΔAHS
    for (int i = 0; i < d; ++i)
        tmpRes.insert(s2[i].second);

    for (auto &p : BN) {
        if (tmpRes.find(p.first) == tmpRes.end()) {
            Tree[rank[p.first]].isBranchNode = false;
            Tree[rank[p.first]].posP2H_3.clear();
        }
    }
    for (auto &p : tmpRes)
        res.push_back(p);
}

void Graph::P2H_3(vector<int> &vIDSet)
{
    if (!isOptimizeVS) {
        optimizeVS();
        isOptimizeVS = true;
    }

    for (auto &vID : vIDSet) {
        int v = rank[vID];
        if (Tree[v].posP2H_3.size() == 0) {
            Tree[v].posP2H_3.resize(nodenum);
        }

        if (Tree[v].ch.size() < 2) {
            cout << "vID=" << vID << "   Tree[v].ch.size: " << Tree[v].ch.size() << endl;
            // return;
        }

        Tree[v].isBranchNode = true;

        vector<int> childID;
        childID.reserve(nodenum);

        queue<int> q;
        q.push(v);
        while (!q.empty()) {
            int top = q.front();
            q.pop();
            for (auto &p : Tree[top].ch) {
                q.push(p);
                childID.push_back(Tree[p].uniqueVertex);
            }
        }
        for (auto &s : childID) {
            vector<int> &posP2H = Tree[v].posP2H_3[s];
            int posOptimizeSize = Tree[v].posOptimize.size();
            s = rank[s];
            for (int c = 0; c < posOptimizeSize; ++c) {
                bool flag = true;
                int d_s_c = Tree[s].dis[Tree[v].posOptimize[c]];

                for (int u = 0; u < posOptimizeSize; ++u) {
                    if (u == c) {
                        continue;
                    }
                    int d_s_u = Tree[s].dis[Tree[v].posOptimize[u]];
                    if (d_s_u > d_s_c)
                        continue;
                    // dis(u,c)
                    int d_u_c = -1;
                    int low = Tree[v].posOptimize[u] > Tree[v].posOptimize[c] ? u : c;
                    int high = u + c - low;

                    if (low == (posOptimizeSize - 1)) {
                        d_u_c = Tree[v].dis[Tree[v].posOptimize[high]];
                    } else {
                        d_u_c = Tree[rank[Tree[v].vertOptimize[low]]].dis[Tree[v].posOptimize[high]];
                    }
                    if (d_s_c == (d_s_u + d_u_c)) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    posP2H.push_back(Tree[v].posOptimize[c]);
                }
            }
        }
    }
}

vector<int> _DD, _DD2;
struct DegComp {
    int x;
    DegComp(int _x)
    {
        x = _x;
    }
    bool operator<(const DegComp d) const
    {
        if (_DD[x] != _DD[d.x])
            return _DD[x] < _DD[d.x];
        if (_DD2[x] != _DD2[x])
            return _DD2[x] < _DD2[d.x];
        return x < d.x;
    }
};

int Graph::LCAQuery(int _p, int _q)
{
    int p = toRMQ[_p], q = toRMQ[_q];
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
    if (Tree[RMQIndex[k][p]].height < Tree[RMQIndex[k][q]].height)
        return RMQIndex[k][p];
    else
        return RMQIndex[k][q];
}

void Graph::GraphConstract(string orderfileWritePath, string notes)
{

    // initialize E
    _DD.assign(nodenum, 0);
    _DD2.assign(nodenum, 0);
    DD.assign(nodenum, 0);
    DD2.assign(nodenum, 0);

    set<DegComp> Deg;
    int degree;
    for (int i = 0; i < nodenum; i++) {
        degree = Neighbor[i].size();
        _DD[i] = degree;
        _DD2[i] = degree;
        DD[i] = degree;
        DD2[i] = degree;
        Deg.insert(DegComp(i));
    }

    // cout<<"kkkkkkkkkkkkkkkkkkkkk"<<endl;

    // order.clear();
    vector<bool> exist;
    exist.assign(nodenum, true);
    vector<bool> change;
    change.assign(nodenum, false);

    vector<pair<int, pair<int, int>>> vect;
    NeighborCon.assign(nodenum, vect); // NeighborCon.clear();
    // SCconNodes.clear();

    // cout<<"Begin to contract"<<endl;
    int count = 0;
    int Twidth = 0; // tree width
    while (!Deg.empty()) {
        if (count % 100000 == 0)
            cout << "count " << count << " , treewidth " << Twidth << endl;
        count += 1;
        int x = (*Deg.begin()).x;

        while (true) {
            if (change[x]) {
                Deg.erase(DegComp(x));
                _DD[x] = DD[x];
                _DD2[x] = DD2[x];
                Deg.insert(DegComp(x));
                change[x] = false;
                x = (*Deg.begin()).x;
            } else
                break;
        }

        vNodeOrder.push_back(x);
        Deg.erase(Deg.begin());
        exist[x] = false;

        vector<pair<int, pair<int, int>>> Neigh; // Neigh.clear();

        for (auto it = E[x].begin(); it != E[x].end(); it++) {
            if (exist[(*it).first]) {
                Neigh.push_back(*it);
            }
        }
        NeighborCon[x].assign(Neigh.begin(), Neigh.end());
        Twidth = Neigh.size() > Twidth ? Neigh.size() : Twidth;

        // multi threads for n^2 combination
        for (int nid = 0; nid < Neigh.size(); nid++) {
            deleteE(Neigh[nid].first, x);
            change[Neigh[nid].first] = true;
        }

        // if(NeighborCon[x].size()==0) cout<<"neighbor size is zero "<<x<<endl;
        if (Neigh.size() <= 1000) {
            for (int i = 0; i < Neigh.size(); i++) {
                for (int j = i + 1; j < Neigh.size(); j++) {
                    insertE(Neigh[i].first, Neigh[j].first, Neigh[i].second.first + Neigh[j].second.first);
                    change[Neigh[i].first] = true;
                    change[Neigh[j].first] = true;
                    if (Neigh[i].first < Neigh[j].first) {
                        SCconNodes[make_pair(Neigh[i].first, Neigh[j].first)].push_back(x); // no direction
                    } else if (Neigh[j].first < Neigh[i].first) {
                        SCconNodes[make_pair(Neigh[j].first, Neigh[i].first)].push_back(x);
                    }
                }
            }
        } else {
#ifdef PARAEELE
            // // multiple thread
            cout << "Graph::CHconsorderMT\t\t:" << Neigh.size() << " tree width >1000 and use multiple thread " << endl;
#pragma omp parallel for num_threads(NUM_THREAD)
            for (int i = 0; i < Neigh.size(); i++) {
                change[Neigh[i].first] = true;

                for (int j = 0; j < Neigh.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    int u = Neigh[i].first;
                    int v = Neigh[j].first;
                    int w = Neigh[i].second.first + Neigh[j].second.first;

                    if (E[u].find(v) == E[u].end()) {
                        E[u].insert(make_pair(v, make_pair(w, 1)));
                        DD[u]++;
                        DD2[u]++;
                    } else {
                        if (E[u][v].first > w) {
                            E[u][v] = make_pair(w, 1);
                        } else if (E[u][v].first == w) {
                            E[u][v].second += 1;
                        }
                    }
                    if (Neigh[i].first < Neigh[j].first) {
                        SCconNodes[make_pair(Neigh[i].first, Neigh[j].first)].push_back(x);
                    }
                }
            }

            // int step = Neigh.size() / threadnum;
            // boost::thread_group thread;
            // for (int i = 0; i < threadnum; i++)
            // {
            // 	pair<int, int> p;
            // 	p.first = i * step;
            // 	if (i == threadnum - 1)
            // 		p.second = Neigh.size();
            // 	else
            // 		p.second = (i + 1) * step;
            // 	thread.add_thread(new boost::thread(&Graph::NeighborComorder, this, boost::ref(Neigh), p, x));
            // }
            // thread.join_all();
#endif
        }
        for (int i = 0; i < Neigh.size(); ++i) {
            int y = Neigh[i].first;
            if ((_DD[y] != DD[y]) || (_DD2[y] != DD2[y])) {
                Deg.erase(DegComp(y));
                _DD[y] = DD[y];
                _DD2[y] = DD2[y];
                Deg.insert(DegComp(y));
                change[y] = false;
            }
        }
    }

    NodeOrder.assign(nodenum, 0);
    for (int k = 0; k < vNodeOrder.size(); k++) {
        NodeOrder[vNodeOrder[k]] = k;
    }
    // cout<<"Finish Contract"<<endl;

    // write the order information
    ofstream OF(orderfileWritePath);
    OF << nodenum << endl;
    for (int nodeID = 0; nodeID < nodenum; nodeID++) {
        OF << nodeID << " " << NodeOrder[nodeID] << endl;
    }
    OF << notes << endl;
    int edgenum = 0, superenum = 0;
    for (int i = 0; i < nodenum; i++) {
        edgenum += Neighbor[i].size();
        superenum += NeighborCon[i].size();
    }
    cout << "edge number " << edgenum << " super edge number " << superenum << " ratio " << (double)superenum / edgenum << endl;
}

void Graph::deleteE(int u, int v)
{
    if (E[u].find(v) != E[u].end()) {
        E[u].erase(E[u].find(v));
        DD[u]--;
    }

    if (E[v].find(u) != E[v].end()) {
        E[v].erase(E[v].find(u));
        DD[v]--;
    }
}
void Graph::insertE(int u, int v, int w)
{
    if (E[u].find(v) == E[u].end()) {
        E[u].insert(make_pair(v, make_pair(w, 1)));
        DD[u]++;
        DD2[u]++;
    } else {
        if (E[u][v].first > w)
            E[u][v] = make_pair(w, 1);
        else if (E[u][v].first == w)
            E[u][v].second += 1;
    }

    if (E[v].find(u) == E[v].end()) {
        E[v].insert(make_pair(u, make_pair(w, 1)));
        DD[v]++;
        DD2[v]++;
    } else {
        if (E[v][u].first > w)
            E[v][u] = make_pair(w, 1);
        else if (E[v][u].first == w)
            E[v][u].second += 1;
    }
}
