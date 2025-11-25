#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


#define TAG_REQUEST 1
#define TAG_WORK    2
#define TAG_STOP    3
#define CHUNK_SIZE 10000

/* Lignes de commande */
// mpirun -np <nb_processus> ./d3-1 N
// mpicc d3-1.c -o d3-1

/* Savoir si n est premier */
int isPrime(long long n) {
    if (n < 2) return 0;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {

    //Initialisation du MPI
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

    long long local_count = 0;
    // Travail du maître
    if (rank == 0) {
        long long start_p = 2;
        long long end_p   = N - 6;
        long long next_p  = start_p;
        int active_workers = size - 1;

        while (active_workers > 0) {
            MPI_Status status;
            MPI_Recv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE,
                        TAG_REQUEST, MPI_COMM_WORLD, &status);

            int worker = status.MPI_SOURCE;

            // plus de travail on arrête les ouvriers
            if (next_p > end_p) {
                MPI_Send(NULL, 0, MPI_CHAR, worker, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
            // On donne la partie suivante à étudier pour les ouvriers
            else {
                long long s = next_p;
                long long e = s + CHUNK_SIZE - 1;
                if (e > end_p) e = end_p;
                next_p = e + 1;

                long long interval[2] = { s, e };

                MPI_Send(interval, 2, MPI_LONG_LONG,
                         worker, TAG_WORK, MPI_COMM_WORLD);
            }
        }
    }

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
