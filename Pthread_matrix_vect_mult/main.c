
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define m 4
#define n 2

int thread_count;
double A[m][n];
double x[n];
double y[m];

void* Pth_mat_vect(void* rank);

int main(int argc, char* argv[])
{
    long thread;
    pthread_t* thread_handles;

    thread_count = strtol(argv[1], NULL, 10);

    thread_handles = malloc(thread_count * sizeof(pthread_t));

    double cont = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i][j] = cont;
            cont += 2.0;
        }
    }

    for (int i = 0; i < n; ++i)
    {
        x[i] = (double) i;
    }

    for (thread = 0; thread < thread_count; ++thread)
    {
        pthread_create(&thread_handles[thread], NULL, Pth_mat_vect, (void*) thread);
    }

    for (thread = 0; thread < thread_count; ++thread)
    {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);

    for (int i = 0; i < m; ++i)
    {
        printf("%f, ", y[i]);
    }
    printf("\n");

    return 0;
}

void* Pth_mat_vect(void* rank)
{
    long my_rank = (long) rank;
    int i, j;
    int local_m = m/thread_count;
    int my_first_row = my_rank*local_m;
    int my_last_row = (my_rank+1)*local_m - 1;

    for (i = my_first_row; i <= my_last_row; ++i)
    {
        y[i] = 0.0;
        for (j = 0; j < n; ++j)
        {
            y[i] += A[i][j] * x[j];
        }
    }

    return NULL;
}