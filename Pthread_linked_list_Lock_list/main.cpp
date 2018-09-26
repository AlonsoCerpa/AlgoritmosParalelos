#include <iostream>
#include <random>
#include <pthread.h>
#include <chrono>


struct list_node_s
{
    int data;
    struct list_node_s* next;
};

int Insert(int value, struct list_node_s** head_p);
int Member(int value, struct list_node_s* head_p);
int Delete(int value, struct list_node_s** head_p);
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
list_node_s* head_ptr;
std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
pthread_mutex_t mutex;


int main(int argc, char* argv[])
{
    int init_keys = 1000;
    long thread_ops;
    pthread_t* thread_handles;

    thread_count = strtol(argv[1], NULL, 10);
    thread_ops = cont_ops / thread_count;
    thread_handles = (pthread_t*) malloc(thread_count * sizeof(pthread_t));

    head_ptr = new list_node_s();
    head_ptr->data = dist(rng);
    head_ptr->next = NULL;
    list_node_s* node_ptr;

    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < init_keys; ++i)
    {
        Insert(dist(rng), &head_ptr);
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

    pthread_mutex_destroy(&mutex);
    free(thread_handles);

    //printList(head_ptr);

    return 0;
}

void* Thread_op(void* arg)
{
    long thread_ops = (long) arg;
    for (int i = 0; i < thread_ops; ++i)
    {
        op = dist1(rng);
        if (op <= cont_member)
        {
            pthread_mutex_lock(&mutex);
            Member(dist(rng), head_ptr);
            pthread_mutex_unlock(&mutex);
        }
        else if (op > cont_member && op <= cont_member + cont_insert)
        {
            pthread_mutex_lock(&mutex);
            Insert(dist(rng), &head_ptr);
            pthread_mutex_unlock(&mutex);
        }
        else
        {
            pthread_mutex_lock(&mutex);
            Delete(dist(rng), &head_ptr);
            pthread_mutex_unlock(&mutex);
        }
    }
}



int Member(int value, struct list_node_s* head_p)
{
    struct list_node_s* curr_p = head_p;

    while (curr_p != NULL && curr_p->data < value)
    {
        curr_p = curr_p->next;
    }

    if (curr_p == NULL || curr_p->data > value)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

int Insert(int value, struct list_node_s** head_p)
{
    struct list_node_s* curr_p = *head_p;
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
            *head_p = temp_p;
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

int Delete(int value, struct list_node_s** head_p)
{
    struct list_node_s* curr_p = *head_p;
    struct list_node_s* pred_p = NULL;

    while (curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if (curr_p != NULL && curr_p->data == value)
    {
        if (pred_p == NULL)
        {
            *head_p = curr_p->next;
            free(curr_p);
        }
        else
        {
            pred_p->next = curr_p->next;
            free(curr_p);
        }
        return 1;
    }
    else
    {
        return 0;
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