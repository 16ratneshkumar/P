# Breadth-First Search
```cpp
const int V = 5;

int graph[V][V] = {
    {0, 1, 1, 0, 0},
    {1, 0, 0, 1, 0},
    {1, 0, 0, 1, 1},
    {0, 1, 1, 0, 0},
    {0, 0, 1, 0, 0}
};

bool visited[V];

void bft(int start) {
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int v = q.front();
        q.pop();
        cout << v << " ";

        for (int u = 0; u < V; u++) {
            if (graph[v][u] == 1 && !visited[u]) {
                visited[u] = true;
                q.push(u);
            }
        }
    }
}

int main() {
    for (int i = 0; i < V; i++)
        visited[i] = false;

    cout << "Breadth First Traversal starting from vertex 0: ";
    bft(0);
    cout << endl;

    return 0;
}

```

# Depth-First Search
```cpp
const int V = 5;

int graph[V][V] = {
    {0, 1, 1, 0, 0},
    {1, 0, 0, 1, 0},
    {1, 0, 0, 1, 1},
    {0, 1, 1, 0, 0},
    {0, 0, 1, 0, 0}
};

bool visited[V];

void dft(int v) {
    visited[v] = true;
    cout << v << " ";
    for (int u = 0; u < V; u++) {
        if (graph[v][u] == 1 && !visited[u]) {
            dft(u);
        }
    }
}

int main() {
    for (int i = 0; i < V; i++)
        visited[i] = false;

    cout << "Depth First Traversal starting from vertex 0: ";
    dft(0);
    cout << endl;

    return 0;
}
```