#include <iostream>
#include <random>

struct list_node_s
{
    int data;
    struct list_node_s* next;
};

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

int main()
{
    int min = 0;
    int max = 1000;
    int cont_insert = 100;
    int cont_inserted = 0;
    std::mt19937 rng;
    //rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);

    list_node_s* head_ptr = new list_node_s();
    head_ptr->data = dist(rng);
    head_ptr->next = NULL;
    list_node_s* node_ptr;

    for (int i = 0; i < cont_insert; ++i)
    {
        Insert(dist(rng), &head_ptr);
        ++cont_inserted;
    }

    printList(head_ptr);

    return 0;
}