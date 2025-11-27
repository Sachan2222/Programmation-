#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define CHUNK_SIZE 10000

/* Vérifie si un nombre est premier */
int isPrime(long long n) {
    if (n < 2) return 0;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 0;
    double start_time = 0.0;
    if (rank == 0) start_time = MPI_Wtime();

    // Diffusion de N
    if (rank == 0) N = atoll(argv[1]);
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (N < 8) {
        if (rank == 0) printf("Nombre de couples sexy <= %lld : 0\n", N);
        MPI_Finalize();
        return 0;
    }

    // Intervalle global et nombre total de p
    long long global_start = 2;
    long long global_end   = N - 6;
    long long total_p      = global_end - global_start + 1;

    // Nombre de chunks
    int num_chunks = (total_p + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Fenêtres RMA
    long long *task_array = NULL;
    long long current_index = 0;
    long long *results = NULL;
    MPI_Win win_tasks, win_index, win_results;

    // Maître prépare les tâches
    if (rank == 0) {
        if (num_chunks > 0) {
            MPI_Alloc_mem(2 * num_chunks * sizeof(long long), MPI_INFO_NULL, &task_array);
            long long p = global_start;
            for (int i = 0; i < num_chunks; i++) {
                long long s = p;
                long long e = p + CHUNK_SIZE - 1;
                if (e > global_end) e = global_end;
                task_array[2*i] = s;
                task_array[2*i + 1] = e;
                p = e + 1;
            }
        }
        MPI_Alloc_mem(size * sizeof(long long), MPI_INFO_NULL, &results);
        memset(results, 0, size * sizeof(long long));

        // Création des fenêtres
        MPI_Win_create(task_array, (num_chunks>0?2*num_chunks*sizeof(long long):0),
                       sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_tasks);
        MPI_Win_create(&current_index, sizeof(long long), sizeof(long long),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win_index);
        MPI_Win_create(results, size*sizeof(long long), sizeof(long long),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win_results);
    } else {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_tasks);
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_index);
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_results);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Toutes les fenêtres sont prêtes

    long long local_count = 0;

    while (1) {
        long long prev_index = -1;
        long long one = 1;

        // Récupérer un chunk de façon atomique
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_index);
        MPI_Fetch_and_op(&one, &prev_index, MPI_LONG_LONG, 0, 0, MPI_SUM, win_index);
        MPI_Win_unlock(0, win_index);

        if (prev_index >= num_chunks) break; // plus de chunks

        long long interval[2];
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win_tasks);
        MPI_Get(interval, 2, MPI_LONG_LONG, 0, 2*prev_index, 2, MPI_LONG_LONG, win_tasks);
        MPI_Win_unlock(0, win_tasks);

        long long s = interval[0];
        long long e = interval[1];

        for (long long p = s; p <= e; p++) {
            long long q = p + 6;
            if (q > N) continue;
            if (isPrime(p) && isPrime(q)) local_count++;
        }
    }

    // Écriture du résultat dans la fenêtre RMA du maître
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_results);
    MPI_Put(&local_count, 1, MPI_LONG_LONG, 0, rank, 1, MPI_LONG_LONG, win_results);
    MPI_Win_unlock(0, win_results);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        long long global_count = 0;
        for (int i = 0; i < size; i++) global_count += results[i];
        printf("Nombre de couples sexy <= %lld : %lld\n", N, global_count);
        double end_time = MPI_Wtime();
        printf("Temps d'exécution : %f secondes\n", end_time - start_time);

        if (task_array) MPI_Free_mem(task_array);
        MPI_Free_mem(results);
    }

    MPI_Win_free(&win_tasks);
    MPI_Win_free(&win_index);
    MPI_Win_free(&win_results);
 
    MPI_Finalize();
    return 0;
}
