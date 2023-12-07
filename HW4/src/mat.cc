#include <stdio.h>
#include <mpi.h>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                   int **a_mat_ptr, int **b_mat_ptr) {
    int world_rank, world_size, a_size, b_size;
    // TODO: init MPI
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0) {
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        printf("n = %d, m = %d, l = %d\n", *n_ptr, *m_ptr, *l_ptr);
    }

    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    a_size = (*n_ptr) * (*m_ptr);
    b_size = (*m_ptr) * (*l_ptr);

    *a_mat_ptr = (int*)calloc(a_size, sizeof(int));
    *b_mat_ptr = (int*)calloc(b_size, sizeof(int));

    if(world_rank == 0) {
        for(int i = 0 ; i < a_size ; i++) {
            scanf("%d", (*a_mat_ptr) + i);
        }
        for(int i = 0 ; i < b_size ; i++) {
            scanf("%d", (*b_mat_ptr) + i);
        }
    }

    MPI_Bcast(*a_mat_ptr, a_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, b_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat) {
    int world_rank, world_size;
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int *C, *ans;
    int start, end;
    int C_size = n * l;
    start = (n / world_size) * world_rank;
    end = (world_rank == world_size-1) ? n : start + (n / world_size);

    C = (int*)calloc(C_size, sizeof(int));
    ans = (int*)calloc(C_size, sizeof(int));

    int i, j, k, a_idx, b_idx, c_idx;
    a_idx = start * m;

    for(i = start ; i < end ; i++, a_idx += m){
        c_idx = i * l;
        b_idx = 0;
        for(j = 0 ; j < l ; j++){
            for(k = 0 ; k < m ; k++){
                C[c_idx] += a_mat[a_idx + k] * b_mat[b_idx];
                b_idx++;
            }
            c_idx++;
        }
    }

    // for(i = 0; i < C_size; i++) {
    //     printf("%d ", C[i]);
    // }
    // printf("\n");
    // printf("Process %d is about to perform MPI_Reduce\n", world_rank);
    MPI_Reduce(C, ans, C_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // printf("Process %d has completed MPI_Reduce\n", world_rank);

    if(world_rank == 0) {
        // printf("Process 0 is about to print the ans array\n");
        for(i = 0 ; i < n ; i++){
            c_idx = i * l;
            for(j = 0 ; j < l ; j++) {
                printf("%d ", ans[c_idx++]);
            }
            printf("\n");
        }
        // printf("\nProcess 0 has completed printing the ans array\n");
    }
}

void destruct_matrices(int *a_mat, int *b_mat) {
    free(a_mat);
    free(b_mat);
}