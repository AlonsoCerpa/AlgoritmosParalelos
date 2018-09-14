#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>

const int M = 4;
const int N = 2;

//mpicc main.c
//mpiexec -n 3 ./a.out

int main(void)
{
	int comm_sz;
	int my_rank;

    double vec[N];
    double* mat;
    double* final_r;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int block_size = (M*N)/comm_sz;
    int res_parc_sz = block_size / N;
    double local_mat[block_size];
    double res_parc[res_parc_sz];

    if (my_rank == 0)
    {
        int cont = 0;
        mat = malloc(M*N*sizeof(double));
        final_r = malloc(M*sizeof(double));
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                mat[(i*N)+j] = cont;
                ++cont;
            }
        }
        for (int j = 0; j < N; ++j)
        {
            vec[j] = j;
        }
    }

    MPI_Bcast(vec, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(mat, block_size, MPI_DOUBLE, local_mat,
                block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < res_parc_sz; ++i)
    {
        res_parc[i] = 0;
        for (int j = 0; j < N; ++j)
        {
            res_parc[i] += local_mat[(i*N)+j] * vec[j];
        }
    }

    MPI_Gather(res_parc, res_parc_sz, MPI_DOUBLE, final_r,
               res_parc_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        for (int i = 0; i < M; ++i)
        {
            printf("%f ", final_r[i]);
        }
        printf("\n");
        free(mat);
        free(final_r);
    }

	MPI_Finalize();
	return 0;
}