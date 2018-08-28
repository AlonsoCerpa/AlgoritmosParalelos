#include <iostream>
#include <vector>
#include <chrono>

template<class T>
std::vector<std::vector<T> > classic_matrix_mult(std::vector<std::vector<T> > &m1, std::vector<std::vector<T> > &m2)
{
    std::vector<std::vector<T> > m_new(m1.size(), std::vector<T>(m2[0].size(), 0));
    
    for (int j = 0; j < m1.size(); ++j)
    {
        for (int h = 0; h < m2[0].size(); ++h)
        {
            for (int i = 0; i < m1[0].size(); ++i)
            {
                m_new[j][h] += m1[j][i] * m2[i][h];
            }
        }
    }

    return m_new;
}

int min(int a, int b)
{
    if (a < b)
        return a;
    else
        return b;
}

template<class T>
std::vector<std::vector<T> > block_matrix_mult(std::vector<std::vector<T> > &m1, std::vector<std::vector<T> > &m2, int b)
{
    std::vector<std::vector<T> > m_new(m1.size(), std::vector<T>(m2[0].size(), 0));

    for (int i0 = 0; i0 < m1.size(); i0+=b)
    {
        for (int j0 = 0; j0 < m1.size(); j0+=b)
        {
            for (int k0 = 0; k0 < m1.size(); k0+=b)
            {
                for (int i = i0; i < min(i0+b, m1.size()); ++i)
                {
                    for (int j = j0; j < min(j0+b, m1.size()); ++j)
                    {
                        for (int k = k0; k < min(k0+b, m1.size()); ++k)
                        {
                            m_new[i][j] += m1[i][k] * m2[k][j];
                        }
                    }
                }
            }
        }
    }

    return m_new;
}

int main()
{
    std::vector<std::vector<int> > m1(1000, std::vector<int>(1000, 1));
    std::vector<std::vector<int> > m2(1000, std::vector<int>(1000, 2));

    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    std::vector<std::vector<int> > m = classic_matrix_mult(m1, m2);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count() << std::endl;

/*
    for (int i = 0; i < m.size(); ++i)
    {
        for (int j = 0; j < m[0].size(); ++j)
        {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";*/

    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    int b = 100;
    std::vector<std::vector<int> > mr = block_matrix_mult(m1, m2, b);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << std::endl;

/*
    for (int i = 0; i < mr.size(); ++i)
    {
        for (int j = 0; j < mr[0].size(); ++j)
        {
            std::cout << mr[i][j] << " ";
        }
        std::cout << "\n";
    }*/

    return 0;
}