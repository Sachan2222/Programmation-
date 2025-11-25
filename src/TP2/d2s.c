#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
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

static void merge_sort_recursive(int *arr, int *tmp, size_t left, size_t right) {
    if (right - left <= 1) {
        return;
    }
    size_t mid = left + (right - left) / 2;
    merge_sort_recursive(arr, tmp, left, mid);
    merge_sort_recursive(arr, tmp, mid, right);
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

    double start = now_seconds();
    if (n > 1) {
        merge_sort_recursive(arr, tmp, 0, n);
    }
    double elapsed = now_seconds() - start;

    if (n > 0) {
        print_output(arr, n);
    }

    fprintf(stderr, "# sequential sort: n=%zu time=%.6f s\n", n, elapsed);

    free(arr);
    free(tmp);
    return EXIT_SUCCESS;
}
