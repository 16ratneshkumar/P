# 0-1 Knapsack
```cpp
int knapsack(int W, int wt[], int val[], int n) {
    int dp[n + 1][W + 1];

    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0)
                dp[i][w] = 0;
            else if (wt[i - 1] <= w)
                dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
            else
                dp[i][w] = dp[i - 1][w];
        }
    }

    return dp[n][W];
}

int main() {
    int val[] = {60, 100, 120};
    int wt[] = {10, 20, 30};
    int W = 50;
    int n = sizeof(val) / sizeof(val[0]);

    cout << knapsack(W, wt, val, n);
    return 0;
}
```

# Prim's Algorithm (Simplest C++ Version with Adjacency Matrix)
```cpp
#include <iostream>
using namespace std;

const int V = 5;
const int INF = 999999;

int minKey(int key[], bool mstSet[]) {
    int min = INF, min_index;
    for (int v = 0; v < V; v++)
        if (!mstSet[v] && key[v] < min)
            min = key[v], min_index = v;
    return min_index;
}

void primMST(int graph[V][V]) {
    int parent[V], key[V];
    bool mstSet[V];

    for (int i = 0; i < V; i++)
        key[i] = INF, mstSet[i] = false;

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet);
        mstSet[u] = true;

        for (int v = 0; v < V; v++)
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }

    for (int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << " : " << graph[i][parent[i]] << endl;
}

int main() {
    int graph[V][V] = {
        {0, 2, 0, 6, 0},
        {2, 0, 3, 8, 5},
        {0, 3, 0, 0, 7},
        {6, 8, 0, 0, 9},
        {0, 5, 7, 9, 0}
    };

    primMST(graph);
    return 0;
}
```

# Strassen's Matrix
```cpp
#include <iostream>
using namespace std;

void strassenMultiply(int A[2][2], int B[2][2], int C[2][2]) {
    int A11 = A[0][0], A12 = A[0][1], A21 = A[1][0], A22 = A[1][1];
    int B11 = B[0][0], B12 = B[0][1], B21 = B[1][0], B22 = B[1][1];

    int P = (A11 + A22) * (B11 + B22);
    int Q = (A21 + A22) * B11;
    int R = A11 * (B12 - B22);
    int S = A22 * (B21 - B11);
    int T = (A11 + A12) * B22;
    int U = (A21 - A11) * (B11 + B12);
    int V = (A12 - A22) * (B21 + B22);

    C[0][0] = P + S - T + V;
    C[0][1] = R + T;
    C[1][0] = Q + S;
    C[1][1] = P + R - Q + U;
}

int main() {
    int A[2][2] = {{1, 2}, {3, 4}};
    int B[2][2] = {{5, 6}, {7, 8}};
    int C[2][2]; // Result matrix

    strassenMultiply(A, B, C);

    cout << "Resultant Matrix (C = A × B):" << endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
            cout << C[i][j] << " ";
        cout << endl;
    }

    return 0;
}
```