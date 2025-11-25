#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


#define CHUNK_SIZE 10000  // plus utilisé, mais tu peux le garder ou l'enlever

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

    // Initialisation et diffusion de N du maître aux ouvriers
    if (rank == 0) {
        N = atoll(argv[1]);
    }
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Pas de couples sexy si N<8
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

    long long *intervals = NULL;   // On va définir le début et la fin de chaque troncon
    MPI_Win win_intervals;

    if (rank == 0) {
        intervals = (long long *) malloc(2 * size * sizeof(long long));
        if (total_p == 0) {
            // Intervalles vides pour tout le monde
            for (int r = 0; r < size; r++) {
                intervals[2*r]     = 1; 
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
                    // intervalle vide
                    intervals[2*r]     = 1;
                    intervals[2*r + 1] = 0;
                }
            }
        }
    }
    ---------------------------------------------------------------------
        // Travail des ouvriers
    else {
        while (1) {
            MPI_Send(NULL, 0, MPI_CHAR, 0, TAG_REQUEST, MPI_COMM_WORLD);

            MPI_Status status;
            long long interval[2];

            MPI_Recv(interval, 2, MPI_LONG_LONG, 0,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP) break;

            long long s = interval[0];
            long long e = interval[1];

            for (long long p = s; p <= e; p++) {
                long long q = p + 6;
                if (q > N) continue;

                if (isPrime(p) && isPrime(q)) {
                    local_count++;
                }
            }
        }
    }

    // Récupération de la valeure finale
    long long global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG,
                MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Nombre de couples sexy <= %lld : %lld\n", N, global_count);
    }

    MPI_Finalize();
    return 0;

}
