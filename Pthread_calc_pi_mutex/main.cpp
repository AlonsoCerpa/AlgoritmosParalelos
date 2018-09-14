#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>

int thread_count;
double sum = 0;
pthread_mutex_t mutex;
long long n = 33554432;

typedef std::chrono::steady_clock::time_point time_pt;

void* Thread_sum(void* rank);

int main(int argc, char* argv[])
{
    long thread;
    pthread_t* thread_handles;

    pthread_mutex_init(&mutex, NULL);

    thread_count = strtol(argv[1], NULL, 10);

    thread_handles = (pthread_t*) malloc(thread_count * sizeof(pthread_t));

    time_pt begin = std::chrono::steady_clock::now();
    for (thread = 0; thread < thread_count; ++thread)
    {
        pthread_create(&thread_handles[thread], NULL, Thread_sum, (void*) thread);
    }

    for (thread = 0; thread < thread_count; ++thread)
    {
        pthread_join(thread_handles[thread], NULL);
    }
    time_pt end = std::chrono::steady_clock::now();

    pthread_mutex_destroy(&mutex);

    sum *= 4;

    free(thread_handles);

    printf("Calculo de pi paralelo\n");
    printf("Tipo: MUTEX\n");
    printf("Pi: %f\n", sum);
    std::cout << "Tiempo: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" <<std::endl;
    printf("n: %lld\n", n);
    printf("Numero de threads: %d\n", thread_count);

    return 0;
}

void* Thread_sum(void* rank)
{
    long my_rank = (long) rank;
    double factor, my_sum = 0.0;
    long long i;
    long long my_n = n/thread_count;
    long long my_first_i = my_n * my_rank;
    long long my_last_i = my_first_i + my_n;

    if (my_first_i % 2 == 0)
        factor = 1.0;
    else
        factor = -1.0;

    for (i = my_first_i; i < my_last_i; ++i, factor = -factor)
    {
        my_sum += factor/(2*i+1);
    }

    pthread_mutex_lock(&mutex);
    sum += my_sum;
    pthread_mutex_unlock(&mutex);

    return NULL;
}
