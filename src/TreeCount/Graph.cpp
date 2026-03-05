#include "Graph.h"

Graph_::Graph_()
{
    n_ = m_ = 0;
    E_.clear();
}

Graph_::Graph_(const char *file) : Graph_() { ReadGraph(file); }

void Graph_::ReadGraph(const char *file)
{
    FILE *fin = fopen(file, "r");
    fscanf(fin, "%d", &n_);
    fscanf(fin, "%d", &m_);

    for (int i = 0; i <= n_; ++i) { // set up the vertices, each vertex
                                    // is a map of (incident vertex v, dis)
        std::map<int, int> v;
        v.clear();
        E_.push_back(v);
    }
    for (int i = 0; i < m_; i++) {
        int u, v; // vertex u, v and weight w
        fscanf(fin, "%d%d", &u, &v);
        assert(u <= n_ && v <= n_);
        E_[u].insert(std::make_pair(v, 1));
        E_[v].insert(std::make_pair(u, 1));
    }
    D_.clear();
    D_.push_back(0); // vertex id starts from 1
    for (int i = 1; i <= n_; ++i)
        D_.push_back(E_[i].size());
    if (fin != NULL) {
        fclose(fin);
        fin = NULL;
    }
    printf("Read Graph_ OK!\nn = %d, m = %d\n", n_, m_);
}

void Graph_::ReadWeightedGraph(const char *file)
{

    FILE *fin = fopen(file, "r");
    fscanf(fin, "%d", &n_);
    fscanf(fin, "%d", &m_);

    for (int i = 0; i <= n_; ++i) { // set up the vertices, each vertex
                                    // is a map of (incident vertex v, dis)
        std::map<int, int> v;
        v.clear();
        E_.push_back(v);
        my_E_.push_back(v);
    }
    for (int i = 0; i < m_; i++) {
        int u, v, w; // vertex u, v and weight w
        fscanf(fin, "%d%d%d", &u, &v, &w);

        // w  = 1;//【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【【

        assert(u <= n_ && v <= n_);
        if (E_[u].find(v) != E_[u].end()) {
            if (E_[u][v] > w)
                E_[u][v] = E_[v][u] = w;
        } else {
            E_[u].insert(std::make_pair(v, w));
            E_[v].insert(std::make_pair(u, w));
        }

        my_E_[u][v] = my_E_[v][u] = 1;
    }
    D_.clear();
    D_.push_back(0); // vertex id starts from 1
    for (int i = 1; i <= n_; ++i)
        D_.push_back(E_[i].size());
    if (fin != NULL) {
        fclose(fin);
        fin = NULL;
    }
    printf("Read Weighted Graph_ OK!\nn = %d, m = %d\n", n_, m_);
}

bool Graph_::isEdgeExist(int u, int v)
{
    return (E_[u].find(v) == E_[u].end()) ? false : true;
}

void Graph_::insertEdge(int u, int v, int w)
{
    if (E_[u].find(v) != E_[u].end())
        return;
    E_[u].insert(std::make_pair(v, w));
    E_[v].insert(std::make_pair(u, w));
    my_E_[u].insert(std::make_pair(v, w));
    my_E_[v].insert(std::make_pair(u, w));
    ++D_[u];
    ++D_[v];
}

void Graph_::deleteEdge(int u, int v)
{
    if (E_[u].find(v) == E_[u].end())
        return;
    E_[u].erase(E_[u].find(v));
    E_[v].erase(E_[v].find(u));

    my_E_[u].erase(my_E_[u].find(v));
    my_E_[v].erase(my_E_[v].find(u));

    --D_[u];
    --D_[v];
}