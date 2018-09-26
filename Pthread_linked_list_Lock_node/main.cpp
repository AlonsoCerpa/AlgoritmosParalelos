#include <iostream>
#include <random>
#include <pthread.h>
#include <chrono>


struct list_node_s
{
    list_node_s()
    {
        pthread_mutex_init(&mutex, NULL);
    }
    ~list_node_s()
    {
        pthread_mutex_destroy(&mutex);
    }

    int data;
    struct list_node_s* next;
    pthread_mutex_t mutex;
};

int Member(int value);
int Insert(int value);
void printList(struct list_node_s* head_p);
void* Thread_op(void* arg);

typedef std::chrono::steady_clock::time_point time_pt;

long thread_count;
int cont_insert = 100;
int cont_delete = 100;
int cont_member = 10000;
int cont_ops = cont_insert + cont_delete + cont_member;
int op;
std::uniform_int_distribution<std::mt19937::result_type> dist1(1, cont_ops);
std::mt19937 rng;
//rng.seed(std::random_device()());
int min = 0;
int max = 10000;
list_node_s* head_p;
std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
pthread_mutex_t head_p_mutex;

int main(int argc, char* argv[])
{
    int init_keys = 1000;
    long thread_ops;
    pthread_t* thread_handles;

    pthread_mutex_init(&head_p_mutex, NULL);

    thread_count = strtol(argv[1], NULL, 10);
    thread_ops = cont_ops / thread_count;
    thread_handles = (pthread_t*) malloc(thread_count * sizeof(pthread_t));

    head_p = new list_node_s();
    head_p->data = dist(rng);
    head_p->next = NULL;
    list_node_s* node_ptr;

    for (int i = 0; i < init_keys; ++i)
    {
        Insert(dist(rng));
    }

    time_pt begin = std::chrono::steady_clock::now();
    for (int thread = 0; thread < thread_count; ++thread)
    {
        pthread_create(&thread_handles[thread], NULL, Thread_op, (void*) thread_ops);
    }

    for (int thread = 0; thread < thread_count; ++thread)
    {
        pthread_join(thread_handles[thread], NULL);
    }
    time_pt end = std::chrono::steady_clock::now();
    std::cout << "Tiempo: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" <<std::endl;
    printf("Numero de threads: %ld\n", thread_count);

    free(thread_handles);
    pthread_mutex_destroy(&head_p_mutex);

    //printList(head_ptr);

    return 0;
}

void* Thread_op(void* arg)
{
    long thread_ops = (long) arg;
    for (int i = 0; i < thread_ops; ++i)
    {
        Member(dist(rng));
    }
}

int Insert(int value)
{
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;
    struct list_node_s* temp_p;

    while (curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if (curr_p == NULL || curr_p->data > value)
    {
        temp_p = (list_node_s*) malloc(sizeof(struct list_node_s));
        temp_p->data = value;
        temp_p->next = curr_p;

        if (pred_p == NULL)
        {
            head_p = temp_p;
        }
        else
        {
            pred_p->next = temp_p;
        }
        return 1;
    }
    else
    {
        return 0;
    }
}


int Member(int value)
{
    struct list_node_s* temp_p;

    pthread_mutex_lock(&head_p_mutex);
    temp_p = head_p;
    while (temp_p != NULL && temp_p->data < value)
    {
        if (temp_p->next != NULL)
            pthread_mutex_lock(&(temp_p->next->mutex));
        if (temp_p == head_p)
            pthread_mutex_unlock(&head_p_mutex);
        pthread_mutex_unlock(&(temp_p->mutex));
        temp_p = temp_p->next;
    }

    if (temp_p == NULL || temp_p->data > value)
    {
        if (temp_p == head_p)
            pthread_mutex_unlock(&head_p_mutex);
        if (temp_p != NULL)
            pthread_mutex_unlock(&(temp_p->mutex));
        return 0;
    }
    else
    {
        if (temp_p == head_p)
            pthread_mutex_unlock(&head_p_mutex);
        pthread_mutex_unlock(&(temp_p->mutex));
        return 1;
    }
}


void printList(struct list_node_s* head_p)
{
    while (head_p != NULL)
    {
        std::cout << head_p->data << " ";
        head_p = head_p->next;
    }    
    std::cout << "\n";
}