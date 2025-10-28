#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//OMP_NUM_THREADS=8

static size_t g_cutoff = 10; //Seuil de la version séquentiel

static void merge(int *arr, int *tmp, size_t left, size_t mid, size_t right) {
    size_t i = left;
    size_t j = mid;
    size_t k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    while (i < mid) {
        tmp[k++] = arr[i++];
    }
    while (j < right) {
        tmp[k++] = arr[j++];
    }
    for (i = left; i < right; ++i) {
        arr[i] = tmp[i];
    }
}

static void merge_sort_recursive(int *arr, int *tmp, size_t left, size_t right) {
    if (right - left <= 1) {
        return;
    }
    size_t mid = left + (right - left) / 2;
    size_t n = right - left;
    if (n >= g_cutoff) {
        // Crée des tâches pour les deux branches si c’est assez gros
        #pragma omp task shared(arr, tmp) if(n >= g_cutoff)
        merge_sort_recursive(arr, tmp, left, mid);

        #pragma omp task shared(arr, tmp) if(n >= g_cutoff)
        merge_sort_recursive(arr, tmp, mid, right);

        #pragma omp taskwait
    } else {
        // Tri récursif séquentiel pour les petits segments
        merge_sort_recursive(arr, tmp, left, mid);
        merge_sort_recursive(arr, tmp, mid, right);
    }
    merge(arr, tmp, left, mid, right);
}

static void print_output(const int *arr, size_t n) {
    if (n <= 1000) {
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) {
                putchar(' ');
            }
            printf("%d", arr[i]);
        }
        putchar('\n');
        return;
    }

    size_t limit = 100;
    for (size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            putchar(' ');
        }
        printf("%d", arr[i]);
    }
    putchar('\n');
    puts("...");
    for (size_t offset = limit; offset > 0; --offset) {
        size_t idx = n - offset;
        if (offset < limit) {
            putchar(' ');
        }
        printf("%d", arr[idx]);
    }
    putchar('\n');
}


int main(void) {
    printf("Coucou_0");
    long long n_input;
    if (scanf("%lld", &n_input) != 1) {
        fprintf(stderr, "Error: could not read n.\n");
        return EXIT_FAILURE;
    }
    if (n_input < 0) {
        fprintf(stderr, "Error: negative n (%lld).\n", n_input);
        return EXIT_FAILURE;
    }
    
    printf("Coucou_1");
    size_t n = (size_t)n_input;
    int *arr = NULL, *tmp = NULL;

    if (n > 0) {
        arr = (int*)malloc(n * sizeof(int));
        tmp = (int*)malloc(n * sizeof(int));
        if (!arr || !tmp) {
            fprintf(stderr, "Error: memory allocation failure (%zu elements).\n", n);
            free(arr); free(tmp);
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < n; ++i) {
            if (scanf("%d", &arr[i]) != 1) {
                fprintf(stderr, "Error: reading element %zu.\n", i);
                free(arr); free(tmp);
                return EXIT_FAILURE;
            }
        }
    }
    printf("Coucou_2");
    double t0 = omp_get_wtime();

    if (n > 1) {
        // Région parallèle : un seul thread lance la tâche racine
        #pragma omp parallel
        {
            #pragma omp single nowait
            merge_sort_recursive(arr, tmp, 0, n);
        }
    }

    double elapsed = omp_get_wtime() - t0;

    if (n > 0) print_output(arr, n);

    // Log vers stderr pour ne pas polluer la sortie du tri
    int threads_used = 1;
    #pragma omp parallel
    {
        #pragma omp single
        threads_used = omp_get_num_threads();
    }
    fprintf(stderr, "# openmp sort: n=%zu threads=%d time=%.6f s\n",
            n, threads_used, elapsed);

    free(arr);
    free(tmp);
    return EXIT_SUCCESS;
}