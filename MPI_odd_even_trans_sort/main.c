#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

//mpicc main.c
//mpiexec -n 4 ./a.out

const int n = 12;

int compare( const void* a, const void* b)
{
     int int_a = * ( (int*) a );
     int int_b = * ( (int*) b );

     if ( int_a == int_b ) return 0;
     else if ( int_a < int_b ) return -1;
     else return 1;
}

int compute_partner(int phase, int my_rank, int comm_sz)
{
    int partner;
    if (phase % 2 == 0)
    {
        if (my_rank % 2 != 0)
        {
            partner = my_rank - 1;
        }
        else
        {
            partner = my_rank + 1;
        }
    }
    else
    {
        if (my_rank % 2 != 0)
        {
            partner = my_rank + 1;
        }
        else
        {
            partner = my_rank - 1;
        }
    }
    if (partner == -1 || partner == comm_sz)
    {
        partner = MPI_PROC_NULL;
    }
    return partner;
}

void merge_low(int local_keys[], int recv_keys[], int local_n)
{
    int m_i, r_i, t_i;
    int temp_keys[local_n];

    m_i = r_i = t_i = 0;

    while (t_i < local_n)
    {
        if (local_keys[m_i] <= recv_keys[r_i])
        {
            temp_keys[t_i] = local_keys[m_i];
            ++t_i; ++m_i;
        }
        else
        {
            temp_keys[t_i] = recv_keys[r_i];
            ++t_i; ++r_i;
        }
    }

    for (m_i = 0; m_i < local_n; ++m_i)
    {
        local_keys[m_i] = temp_keys[m_i];
    }
}

void merge_high(int local_keys[], int recv_keys[], int local_n)
{
    int m_i, r_i, t_i;
    int temp_keys[local_n];

    m_i = r_i = t_i = local_n - 1;

    while (t_i >= 0)
    {
        if (local_keys[m_i] >= recv_keys[r_i])
        {
            //printf("%d", local_keys[m_i]);
            temp_keys[t_i] = local_keys[m_i];
            --t_i; --m_i;
        }
        else
        {
            temp_keys[t_i] = recv_keys[r_i];
            --t_i; --r_i;
        }
    }

    for (m_i = 0; m_i < local_n; ++m_i)
    {
        //printf("%d", temp_keys[m_i]);
        local_keys[m_i] = temp_keys[m_i];
        //printf("%d", local_keys[m_i]);
    }
}

int main(void)
{
    int comm_sz;
	int my_rank, partner;
    int *keys;
    
    MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int local_n = n/comm_sz;
    int local_keys[local_n];
    int recv_keys[local_n];

    if (my_rank == 0)
    {
        keys = malloc(n*sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < n; ++i)
        {
            keys[i] = rand() % 10;
            printf("%d ", keys[i]);
        }
        printf("\n");
    }
    MPI_Scatter(keys, local_n, MPI_INT, local_keys, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    qsort(local_keys, local_n, sizeof(int), compare);
    
    for (int phase = 0; phase < comm_sz; ++phase)
    {
        partner = compute_partner(phase, my_rank, comm_sz);
        MPI_Sendrecv(local_keys, local_n, MPI_INT, partner, 0, recv_keys, local_n, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (partner != MPI_PROC_NULL)
        {
            if (my_rank < partner)
            {
                merge_low(local_keys, recv_keys, local_n);
            }
            else
            {
                merge_high(local_keys, recv_keys, local_n);
            }
        }
    }

    MPI_Gather(local_keys, local_n, MPI_INT, keys, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            printf("%d ", keys[i]);
        }
        printf("\n");
        free(keys);
    }
    

    MPI_Finalize();
}