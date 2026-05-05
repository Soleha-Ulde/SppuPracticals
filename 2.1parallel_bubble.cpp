#include <iostream>
#include <omp.h>
using namespace std;

// Parallel Bubble Sort (Odd-Even Sort)
void parallel_bubble(int arr[], int n)
{
    for (int i = 0; i < n; i++)
    {
        int first = i % 2;   // even or odd phase

        #pragma omp parallel for
        for (int j = first; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                int t = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = t;
            }
        }
    }
}

int main()
{
    int n;
    cout << "Enter total number of elements: ";
    cin >> n;

    int *arr = new int[n];

    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    // start time
    double start = omp_get_wtime();

    parallel_bubble(arr, n);

    // end time
    double end = omp_get_wtime();

    cout << "\nSorted array:\n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";

    cout << "\nExecution Time (Parallel): " << end - start << " seconds\n";

    delete[] arr;
    return 0;
}

/*
=========== HOW TO RUN ===========

1. Save file:
   parallel_bubble.cpp

2. Compile (OpenMP required):
   g++ -fopenmp parallel_bubble.cpp -o pb

3. Run:
   Windows:
       ./pb

4. Sample Input:
   5
   5 3 1 4 2

5. Output:
   1 2 3 4 5

=================================
*/
