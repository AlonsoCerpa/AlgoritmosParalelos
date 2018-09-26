#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>

#define MAX 100

//Compilar: g++ main.cpp -pthread
//Ejecutar: ./a.out < text.txt 4

void* Tokenize(void* rank);

int thread_count;
sem_t* sems;

int main(int argc, char* argv[])
{
    long thread;
    pthread_t* thread_handles;

    thread_count = strtol(argv[1], NULL, 10);
    thread_handles = (pthread_t*) malloc(thread_count * sizeof(pthread_t));
    sems = (sem_t*) malloc(thread_count * sizeof(sem_t));
    for (int i = 0; i < thread_count; ++i)
    {
        sem_init(&sems[i], 0, 0);
    }
    sem_post(&sems[0]);

    for (thread = 0; thread < thread_count; ++thread)
    {
        pthread_create(&thread_handles[thread], NULL, Tokenize, (void*) thread);
    }

    for (thread = 0; thread < thread_count; ++thread)
    {
        pthread_join(thread_handles[thread], NULL);
    }

    for (int i = 0; i < thread_count; ++i)
    {
        sem_destroy(&sems[i]);
    }

    free(thread_handles);
    free(sems);
}

void* Tokenize(void* rank)
{
    long my_rank = (long) rank;
    int count;
    int next = (my_rank + 1) % thread_count;
    char *fg_rv;
    char my_line[MAX];
    char *my_string;
    char *save_ptr;

    sem_wait(&sems[my_rank]);
    fg_rv = fgets(my_line, MAX, stdin);
    sem_post(&sems[next]);
    while (fg_rv != NULL)
    {
        printf("Thread %ld > my line = %s", my_rank, my_line);

        count = 0;
        my_string = strtok_r(my_line, " \t\n", &save_ptr);
        while (my_string != NULL)
        {
            ++count;
            printf("Thread %ld > string %d = %s\n", my_rank, count, my_string);
            my_string = strtok_r(NULL, " \t\n", &save_ptr);
        }

        sem_wait(&sems[my_rank]);
        fg_rv = fgets(my_line, MAX, stdin);
        sem_post(&sems[next]);
    }

    return NULL;
}