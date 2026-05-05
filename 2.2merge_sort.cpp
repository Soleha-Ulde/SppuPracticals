#include<iostream>
#include<stdlib.h>
#include<omp.h>
using namespace std;

void mergesort(int a[], int i, int j);
void merge(int a[], int i1, int j1, int i2, int j2);

void mergesort(int a[], int i, int j)
{
    int mid;
    if(i < j)
    {
        mid = (i + j) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergesort(a, i, mid);

            #pragma omp section
            mergesort(a, mid + 1, j);
        }

        merge(a, i, mid, mid + 1, j);
    }
}

void merge(int a[], int i1, int j1, int i2, int j2)
{
    int temp[1000];
    int i = i1, j = i2, k = 0;

    while(i <= j1 && j <= j2)
    {
        if(a[i] < a[j])
            temp[k++] = a[i++];
        else
            temp[k++] = a[j++];
    }

    while(i <= j1)
        temp[k++] = a[i++];

    while(j <= j2)
        temp[k++] = a[j++];

    for(i = i1, j = 0; i <= j2; i++, j++)
        a[i] = temp[j];
}

int main()
{
    int *a, n, i;
    double start, stop;

    cout << "Enter total number of elements: ";
    cin >> n;

    a = new int[n];

    cout << "Enter elements: ";
    for(i = 0; i < n; i++)
        cin >> a[i];

    start = omp_get_wtime();

    mergesort(a, 0, n - 1);

    stop = omp_get_wtime();

    cout << "Sorted array is:\n";
    for(i = 0; i < n; i++)
        cout << a[i] << " ";

    cout << "\nExecution Time: " << (stop - start);

    return 0;
}
/*
STEPS TO RUN ON UBUNTU:
optional
1. Install compiler:
   sudo apt update
   sudo apt install g++

2. Save file:
   mergesort.cpp

3. Compile with OpenMP:
   g++ -fopenmp mergesort.cpp -o mergesort

4. Run program:
   ./mergesort

5. (Optional) Set threads:
   export OMP_NUM_THREADS=4
   ./mergesort
   
👉 Example run:

Enter total number of elements: 5
Enter elements: 5 2 9 1 6

👉 Output:

Sorted array is:
1 2 5 6 9
Execution Time: 0.000123
*/
