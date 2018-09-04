#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int N = 6000;

//mpicc main.c
//mpiexec -n 6 ./a.out

int get_parc_sum(int *vec, int step)
{
    int parc_sum = 0;
    for (int i = 0; i < step; ++i)
    {
        parc_sum += vec[i];
    }
    return parc_sum;
}

int main(void)
{
	int vec[N];
	int comm_sz;
	int my_rank;
    int parc_sum;
    int total_sum;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int step = (int)(N / comm_sz);

	if (my_rank == 0)
	{
		for (int i = 0; i < N; ++i)
        {
            vec[i] = i;
        }
        for (int i = 1; i < comm_sz; ++i)
        {
            MPI_Send(&vec[i * step], step, MPI_INT, i, 0, MPI_COMM_WORLD);     
        }
        parc_sum = get_parc_sum(vec, step);
        total_sum = parc_sum;
        for (int i = 1; i < comm_sz; ++i)
        {
            MPI_Recv(&parc_sum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += parc_sum;
        }
        printf("Total sum = %d\n", total_sum);
	}
	else
	{
	    MPI_Recv(vec, step, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        parc_sum = get_parc_sum(vec, step);
        MPI_Send(&parc_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  
	}

	MPI_Finalize();
	return 0;
}