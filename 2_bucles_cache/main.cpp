#include <iostream>
#include <chrono>
#include <random>

int main()
{
    const int MAX = 1000;
    double A[MAX][MAX], x[MAX], y[MAX];
    
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(0, 100);

    for (int idx1 = 0; idx1 < MAX; ++idx1)
    {
        for (int idx2 = 0; idx2 < MAX; ++idx2)
        {
            A[idx1][idx2] = uni(rng);
        }
    }

    for (int idx1 = 0; idx1 < MAX; ++idx1)
    {
        x[idx1] = uni(rng); 
    }

    for (int idx1 = 0; idx1 < MAX; ++idx1)
    {
        y[idx1] = uni(rng);
    }

    int i, j;
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    for (i = 0; i < MAX; i++)
        for (j = 0; j < MAX; j++)
            y[i] += A[i][j] * x[j];
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "Time difference1 = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count() <<std::endl;

    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    for (j = 0; j < MAX; j++)
        for (i = 0; i < MAX; i++)
            y[i] += A[i][j] * x[j];
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "Time difference2 = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() <<std::endl;

    return 0;
}