#include <iostream>
#include <omp.h>
#include <climits>
using namespace std;

// Minimum
void findMin(int a[], int n)
{
    int min_val = INT_MAX;

    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0; i < n; i++)
        if(a[i] < min_val)
            min_val = a[i];

    cout << "Min: " << min_val << endl;
}

// Maximum
void findMax(int a[], int n)
{
    int max_val = INT_MIN;

    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0; i < n; i++)
        if(a[i] > max_val)
            max_val = a[i];

    cout << "Max: " << max_val << endl;
}

// Sum
void findSum(int a[], int n)
{
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++)
        sum += a[i];

    cout << "Sum: " << sum << endl;
}

// Average
void findAvg(int a[], int n)
{
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++)
        sum += a[i];

    cout << "Avg: " << (double)sum/n << endl;
}

int main()
{
    int n;
    cin >> n;

    int a[n];

    for(int i = 0; i < n; i++)
        cin >> a[i];

    findMin(a, n);
    findMax(a, n);
    findSum(a, n);
    findAvg(a, n);

    return 0;
}

/*
=========== HOW TO RUN ===========

1. Save file:
   reduction.cpp

2. Compile:
   g++ -fopenmp reduction.cpp -o red

3. Run:
   Windows:
       ./red

4. Sample Input:
   5
   1 2 3 4 5

5. Output:
   Min: 1
   Max: 5
   Sum: 15
   Avg: 3

=================================
*/
