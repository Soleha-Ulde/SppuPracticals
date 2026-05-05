//new updated code 

#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

const int MAX = 100000;

vector<int> graph[MAX];
bool visited[MAX];

void dfs(int start) {
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int curr;

        // safely pop
        #pragma omp critical
        {
            curr = s.top();
            s.pop();
        }

        // check and mark visited
        if (!visited[curr]) {
            #pragma omp critical
            {
                if (!visited[curr]) {
                    visited[curr] = true;
                    cout << curr << " ";
                }
            }

            // explore neighbors in parallel
            int size = graph[curr].size();

            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                int next = graph[curr][i];

                if (!visited[next]) {
                    #pragma omp critical
                    {
                        s.push(next);
                    }
                }
            }
        }
    }
}

int main() {
    int n, m, start;

    cout << "Enter number of nodes, edges and start node: ";
    cin >> n >> m >> start;

    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;

        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // initialize visited
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    cout << "DFS Traversal: ";
    dfs(start);

    return 0;
}

/*
How to Compile and Run:

1. Save the file:
   ParallelDFS.cpp

2. Compile with OpenMP:
   g++ -fopenmp ParallelDFS.cpp -o dfs

3. Run the program:
   ./dfs  

4. Sample Input:
   Enter number of nodes, edges and start node:
   5 4 0
   0 1
   0 2
   1 3
   1 4

5. Sample Output (may vary due to parallel execution):
   DFS Traversal: 0 2 1 4 3

Notes:
- OpenMP is used to parallelize neighbor exploration.
- Critical sections are used to prevent race conditions.
- Output order may vary due to parallel execution.
- This implementation is for learning purposes and not fully optimized for performance.
*/
