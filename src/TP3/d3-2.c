#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK_SIZE 10000

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

    double start_time = 0.0;
    if (rank == 0) start_time = MPI_Wtime();

    long long N = 0;
    if (rank == 0) N = atoll(argv[1]);
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Cas séquentiel
    if (size == 1) {
        long long count = 0;
        for (long long p = 2; p <= N - 6; p++) {
            long long q = p + 6;
            if (isPrime(p) && isPrime(q)) count++;
        }
        printf("Nombre de couples sexy <= %lld : %lld\n", N, count);
        double end_time = MPI_Wtime();
        printf("Temps d'exécution (séquentiel) : %f secondes\n", end_time - start_time);
        MPI_Finalize();
        return 0;
    }

    if (N < 8) {
        if (rank == 0) printf("Nombre de couples sexy <= %lld : 0\n", N);
        MPI_Finalize();
        return 0;
    }

    // ============================================================
    // Préparation du pool de tâches
    // ============================================================
    long long global_start = 2;
    long long global_end = N - 6;
    long long total_p = global_end - global_start + 1;
    int num_chunks = (total_p + CHUNK_SIZE - 1) / CHUNK_SIZE;

    long long *task_array = NULL;
    long long current_index = 0;
    MPI_Win win_tasks, win_index;

    if (rank == 0) {
        // Créer le pool de chunks
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

        MPI_Win_create(task_array, 
                       (num_chunks > 0) ? 2*num_chunks*sizeof(long long) : 0,
                       sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_tasks);
        MPI_Win_create(&current_index, sizeof(long long), sizeof(long long),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win_index);
    } else {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_tasks);
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win_index);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ============================================================
    // Boucle de récupération dynamique des chunks
    // ============================================================
    long long local_count = 0;

    while (1) {
        long long my_chunk_index;
        long long one = 1;

        // ✅ OPTIMISATION : Combiner lock + fetch + unlock en une seule epoch
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_index);
        MPI_Fetch_and_op(&one, &my_chunk_index, MPI_LONG_LONG, 0, 0, MPI_SUM, win_index);
        MPI_Win_unlock(0, win_index);

        // Vérifier si on a dépassé le nombre de chunks
        if (my_chunk_index >= num_chunks) break;

        // ✅ OPTIMISATION : Récupérer le chunk avec un seul lock
        long long interval[2];
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win_tasks);
        MPI_Get(interval, 2, MPI_LONG_LONG, 0, 2*my_chunk_index, 2, MPI_LONG_LONG, win_tasks);
        MPI_Win_unlock(0, win_tasks);

        long long s = interval[0];
        long long e = interval[1];

        // Calcul du chunk (identique à d3-1)
        for (long long p = s; p <= e; p++) {
            long long q = p + 6;
            if (q > N) continue;
            if (isPrime(p) && isPrime(q)) local_count++;
        }
    }

int *all_counts = NULL;

if (rank == 0) {
    all_counts = malloc(size * sizeof(int));
    memset(all_counts, 0, size * sizeof(int));
}

MPI_Win win;
MPI_Win_create(all_counts,
               (rank == 0 ? size * sizeof(int) : 0),
               sizeof(int),
               MPI_INFO_NULL,
               MPI_COMM_WORLD,
               &win);

MPI_Win_fence(0, win);

/* Chaque processus écrit son résultat dans la fenêtre du maître */
int local_int = (int)local_count;
MPI_Put(&local_int,
        1, MPI_INT,
        0,
        rank,
        1, MPI_INT,
        win);

MPI_Win_fence(0, win);

int global_count = 0;

if (rank == 0) {
    for (int i = 0; i < size; i++) {
        global_count += all_counts[i];
    }

    printf("Nombre de couples sexy <= %lld : %d\n", N, global_count);
    double end_time = MPI_Wtime();
    printf("Temps d'exécution : %f secondes\n", end_time - start_time);
}

MPI_Win_free(&win);

if (rank == 0) free(all_counts);

    MPI_Win_free(&win_tasks);
    MPI_Win_free(&win_index);

    MPI_Finalize();
    return 0;
}