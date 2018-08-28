#include <iostream>
#include <chrono>

int main()
{
    const int MAX = 1000;
    double A[MAX][MAX], x[MAX], y[MAX];

    int i, j;
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    for (i = 0; i < MAX; i++)
        for (j = 0; j < MAX; j++)
            y[i] += A[i][j] * x[j];
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "Time difference1 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count() <<std::endl;

    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    for (j = 0; j < MAX; j++)
        for (i = 0; i < MAX; i++)
            y[i] += A[i][j] * x[j];
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "Time difference2 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() <<std::endl;

    return 0;
}