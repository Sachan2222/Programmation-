#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    size_t max_threads;
    size_t active_threads;
    pthread_mutex_t mutex;
} ThreadControl;

typedef struct {
    int *arr;
    int *tmp;
    size_t left;
    size_t right;
    ThreadControl *ctrl;
} SortTask;

static const size_t PARALLEL_THRESHOLD = 1u << 14; /* bascule en séquentiel sous cette taille */

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void thread_control_init(ThreadControl *ctrl, size_t max_threads) {
    ctrl->max_threads = max_threads > 0 ? max_threads : 1;
    ctrl->active_threads = 1; /* le thread principal compte */
    if (pthread_mutex_init(&ctrl->mutex, NULL) != 0) {
        perror("pthread_mutex_init");
        exit(EXIT_FAILURE);
    }
}

static void thread_control_destroy(ThreadControl *ctrl) {
    pthread_mutex_destroy(&ctrl->mutex);
}

static int thread_control_try_acquire(ThreadControl *ctrl) {
    int ok = 0;
    if (pthread_mutex_lock(&ctrl->mutex) != 0) {
        perror("pthread_mutex_lock");
        exit(EXIT_FAILURE);
    }
    if (ctrl->active_threads < ctrl->max_threads) {
        ctrl->active_threads++;
        ok = 1;
    }
    if (pthread_mutex_unlock(&ctrl->mutex) != 0) {
        perror("pthread_mutex_unlock");
        exit(EXIT_FAILURE);
    }
    return ok;
}

static void thread_control_release(ThreadControl *ctrl) {
    if (pthread_mutex_lock(&ctrl->mutex) != 0) {
        perror("pthread_mutex_lock");
        exit(EXIT_FAILURE);
    }
    ctrl->active_threads--;
    if (pthread_mutex_unlock(&ctrl->mutex) != 0) {
        perror("pthread_mutex_unlock");
        exit(EXIT_FAILURE);
    }
}

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

static void merge_sort_sequential(int *arr, int *tmp, size_t left, size_t right) {
    if (right - left <= 1) {
        return;
    }
    size_t mid = left + (right - left) / 2;
    merge_sort_sequential(arr, tmp, left, mid);
    merge_sort_sequential(arr, tmp, mid, right);
    merge(arr, tmp, left, mid, right);
}

static void parallel_merge_sort_impl(int *arr, int *tmp, size_t left, size_t right, ThreadControl *ctrl);

static void *parallel_worker(void *arg) {
    SortTask *task = (SortTask *)arg;
    parallel_merge_sort_impl(task->arr, task->tmp, task->left, task->right, task->ctrl);
    thread_control_release(task->ctrl);
    free(task);
    return NULL;
}

static void parallel_merge_sort_impl(int *arr, int *tmp, size_t left, size_t right, ThreadControl *ctrl) {
    size_t size = right - left;
    if (size <= 1) {
        return;
    }
    if (size <= PARALLEL_THRESHOLD || ctrl->max_threads == 1) {
        merge_sort_sequential(arr, tmp, left, right);
        return;
    }

    size_t mid = left + size / 2;
    pthread_t thread;
    int spawned = 0;

    if (thread_control_try_acquire(ctrl)) {
        SortTask *task = malloc(sizeof(*task));
        if (task == NULL) {
            thread_control_release(ctrl);
        } else {
            task->arr = arr;
            task->tmp = tmp;
            task->left = left;
            task->right = mid;
            task->ctrl = ctrl;
            int rc = pthread_create(&thread, NULL, parallel_worker, task);
            if (rc != 0) {
                fprintf(stderr, "pthread_create failed: %s\n", strerror(rc));
                free(task);
                thread_control_release(ctrl);
            } else {
                spawned = 1;
            }
        }
    }

    parallel_merge_sort_impl(arr, tmp, mid, right, ctrl);

    if (!spawned) {
        parallel_merge_sort_impl(arr, tmp, left, mid, ctrl);
    } else {
        pthread_join(thread, NULL);
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

static size_t parse_thread_count(int argc, char **argv) {
    if (argc < 2) {
        return 4; /* valeur par défaut raisonnable */
    }
    char *endptr = NULL;
    long value = strtol(argv[1], &endptr, 10);
    if (endptr == argv[1] || *endptr != '\0') {
        fprintf(stderr, "Warning: '%s' is not an integer, falling back to 4 threads.\n", argv[1]);
        return 4;
    }
    if (value < 1) {
        fprintf(stderr, "Warning: thread count < 1, falling back to 1.\n");
        return 1;
    }
    return (size_t)value;
}

int main(int argc, char **argv) {
    size_t max_threads = parse_thread_count(argc, argv);

    long long n_input;
    if (scanf("%lld", &n_input) != 1) {
        fprintf(stderr, "Error: could not read n.\n");
        return EXIT_FAILURE;
    }
    if (n_input < 0) {
        fprintf(stderr, "Error: negative n (%lld).\n", n_input);
        return EXIT_FAILURE;
    }
    size_t n = (size_t)n_input;

    int *arr = NULL;
    int *tmp = NULL;

    if (n > 0) {
        arr = malloc(n * sizeof(int));
        tmp = malloc(n * sizeof(int));
        if (!arr || !tmp) {
            fprintf(stderr, "Error: memory allocation failure (%zu elements).\n", n);
            free(arr);
            free(tmp);
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < n; ++i) {
            if (scanf("%d", &arr[i]) != 1) {
                fprintf(stderr, "Error: reading element %zu.\n", i);
                free(arr);
                free(tmp);
                return EXIT_FAILURE;
            }
        }
    }

    ThreadControl ctrl;
    thread_control_init(&ctrl, max_threads);

    double start = now_seconds();
    if (n > 1) {
        parallel_merge_sort_impl(arr, tmp, 0, n, &ctrl);
    }
    double elapsed = now_seconds() - start;

    if (n > 0) {
        print_output(arr, n);
    }

    fprintf(stderr, "# pthread sort: n=%zu threads=%zu time=%.6f s\n", n, max_threads, elapsed);

    thread_control_destroy(&ctrl);
    free(arr);
    free(tmp);
    return EXIT_SUCCESS;
}
