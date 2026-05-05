//Easy code new updated

#include <iostream>
#include <queue>
#include <omp.h>
using namespace std;

class Node {
public:
    int data;
    Node *left, *right;

    Node(int val) {
        data = val;
        left = right = NULL;
    }
};

// Insert node (level order)
Node* insert(Node* root, int data) {
    if (!root) return new Node(data);

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* temp = q.front();
        q.pop();

        if (!temp->left) {
            temp->left = new Node(data);
            return root;
        } else q.push(temp->left);

        if (!temp->right) {
            temp->right = new Node(data);
            return root;
        } else q.push(temp->right);
    }
    return root;
}

// Parallel BFS (simple)
void bfs(Node* root) {
    if (!root) return;

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();

        // process one level in parallel
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            Node* curr;

            #pragma omp critical
            {
                curr = q.front();
                q.pop();
                cout << curr->data << " ";
            }

            #pragma omp critical
            {
                if (curr->left) q.push(curr->left);
                if (curr->right) q.push(curr->right);
            }
        }
    }
}

int main() {
    Node* root = NULL;
    int n, x;

    cout << "Enter number of nodes: ";
    cin >> n;

    for (int i = 0; i < n; i++) {
        cin >> x;
        root = insert(root, x);
    }

    cout << "BFS Traversal: ";
    bfs(root);

    return 0;
}

/*
How to Compile and Run:

1. Save the file:
   bfs_parallel.cpp

2. Compile using OpenMP:
   g++ -fopenmp bfs_parallel.cpp -o bfs

3. Run the program:
   ./bfs     

4. Sample Input:
   Enter number of nodes: 5
   1 2 3 4 5

5. Output:
   BFS Traversal: 1 2 3 4 5

Note:
- OpenMP is used for parallel processing.
- Critical sections are used to avoid race conditions.
- This implementation is for learning purposes and may not give high performance due to synchronization overhead.
*/
