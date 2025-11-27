#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

/* Lignes de commande */
// mpicc d3-2.c -o d3-2
// mpirun -np <nb_processus> ./d3-2 N

/* Savoir si n est premier */
int isPrime(long long n) {
    if (n < 2) return 0;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {

    // Initialisation du MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    long long N = 0;
    double start_time = 0.0;
    // Initialisation et diffusion de N du maître aux ouvriers
    if (rank == 0) {
        start_time = MPI_Wtime();
        N = atoll(argv[1]);
    }
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (N < 8) {
        if (rank == 0) printf("Nombre de couples sexy <= %lld : 0\n", N);
        MPI_Finalize();
        return 0;
    }

    // Intervalle global de p : [2 .. N-6]
    long long global_start = 2;
    long long global_end   = N - 6;
    long long total_p = (global_end >= global_start)
                            ? (global_end - global_start + 1)
                            : 0;

    long long *intervals = NULL;
    long long *results   = NULL;
    MPI_Win win_intervals, win_results;

    if (rank == 0) {
        MPI_Alloc_mem(2 * size * sizeof(long long), MPI_INFO_NULL, &intervals);
        MPI_Win_create(intervals, 2 * size * sizeof(long long), sizeof(long long),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win_intervals);

        MPI_Alloc_mem(size * sizeof(long long), MPI_INFO_NULL, &results);
        memset(results, 0, size * sizeof(long long)); // Initialise à zéro
        MPI_Win_create(results, size * sizeof(long long), sizeof(long long),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win_results);

        if (total_p == 0) {
            for (int r = 0; r < size; r++) {
                intervals[2*r]       = 1;
                intervals[2*r + 1] = 0;
            }
        } else {
            long long base = total_p / size;
            long long rest = total_p % size;

            long long offset = 0;
            for (int r = 0; r < size; r++) {
                long long count_r = (r < rest) ? (base + 1) : base;
                if (count_r > 0) {
                    long long s = global_start + offset;
                    long long e = s + count_r - 1;
                    intervals[2*r]     = s;
                    intervals[2*r + 1] = e;
                    offset += count_r;
                } else {
                    intervals[2*r]     = 1;
                    intervals[2*r + 1] = 0;
                }
            }
        }

    } else {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_intervals);
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_results);
    }
    MPI_Win_fence(0, win_intervals);
    MPI_Win_fence(0, win_results);

    long long local_count = 0;
    long long s, e;

    if (size > 1) {
        MPI_Get(&s, 1, MPI_LONG_LONG, 0, 2*rank, 1, MPI_LONG_LONG, win_intervals);
        MPI_Get(&e, 1, MPI_LONG_LONG, 0, 2*rank + 1, 1, MPI_LONG_LONG, win_intervals);
    }
    if (rank == 0) {
        s = intervals[0];
        e = intervals[1];
    }

    MPI_Win_fence(0, win_intervals);

    if (s <= e) {
        for (long long p = s; p <= e; p++) {
            long long q = p + 6;
            if (q > N) continue;

            if (isPrime(p) && isPrime(q)) {
                local_count++;
            }
        }
    }

    MPI_Put(&local_count, 1, MPI_LONG_LONG, 0, rank, 1, MPI_LONG_LONG, win_results);


    MPI_Win_fence(0, win_results);

    long long global_count = 0;
    if (rank == 0) {

        for (int r = 0; r < size; r++) {
            global_count += results[r];
        }

        printf("Nombre de couples sexy <= %lld : %lld\n", N, global_count);

        MPI_Free_mem(intervals);
        MPI_Free_mem(results);
    }

    MPI_Win_free(&win_results);

    MPI_Barrier(MPI_COMM_WORLD); // S'assure que tous les processus ont fini
    if (rank == 0) {
        double end_time = MPI_Wtime();
        printf("Temps d'exécution : %f secondes\n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}
