#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int MAX_STRING = 100;

//mpicc main.c
//mpiexec -n 2 ./a.out

int main(void)
{
	int number;
	int comm_sz;
	int my_rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank == 0)
	{
		number = 0;
		MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		while (1)
		{
			MPI_Recv(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			++number;
			MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		}
	}
	else if (my_rank == 1)
	{
		while (1)
		{
			MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			++number;
			MPI_Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();
	return 0;
}