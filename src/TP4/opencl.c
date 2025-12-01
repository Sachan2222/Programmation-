// main.c
// Implémentation de Floyd–Warshall en OpenCL pour le graphe décrit dans l'énoncé.

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CL(err, msg) \
    do { \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error %d: %s\n", err, msg); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel OpenCL (Floyd–Warshall, une étape k)
static const char *kernelSource =
"__kernel void floyd_step(__global int *dist, int k, int n)          \n"
"{                                                                   \n"
"    int i = get_global_id(0);                                       \n"
"    int j = get_global_id(1);                                       \n"
"    if (i >= n || j >= n) return;                                   \n"
"    int ik = dist[i*n + k];                                         \n"
"    int kj = dist[k*n + j];                                         \n"
"    int ij = dist[i*n + j];                                         \n"
"    int cand = ik + kj;                                             \n"
"    if (cand < ij)                                                  \n"
"        dist[i*n + j] = cand;                                       \n"
"}                                                                   \n";

int main(void)
{
    int n;
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "Erreur : impossible de lire n.\n");
        return EXIT_FAILURE;
    }
    if (n <= 0) {
        fprintf(stderr, "Erreur : n doit être > 0.\n");
        return EXIT_FAILURE;
    }

    // -----------------------------
    // 1. Allocation et initialisation du graphe
    // -----------------------------
    int *dist = (int *)malloc(n * n * sizeof(int));
    if (!dist) {
        fprintf(stderr, "Erreur : malloc a échoué.\n");
        return EXIT_FAILURE;
    }

    const int INF = n + 1;  // valeur des arcs inexistants

    // Initialisation: tous les arcs à INF, diagonale à 0
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                dist[i * n + j] = 0;
            else
                dist[i * n + j] = INF;
        }
    }

    // Arcs (i -> i+1) de longueur 1 pour i < n-1
    for (int i = 0; i < n - 1; ++i) {
        dist[i * n + (i + 1)] = 1;
    }

    // Arc (n-1 -> 0) de longueur 1
    dist[(n - 1) * n + 0] = 1;

    // -----------------------------
    // 2. Initialisation OpenCL
    // -----------------------------
    cl_int err;

    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_CL(err, "clGetPlatformIDs (count)");

    if (numPlatforms == 0) {
        fprintf(stderr, "Erreur : aucune plateforme OpenCL trouvée.\n");
        free(dist);
        return EXIT_FAILURE;
    }

    cl_platform_id *platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_CL(err, "clGetPlatformIDs (list)");

    cl_platform_id platform = platforms[0];  // on prend la première
    free(platforms);

    // Recherche d'un device GPU, sinon CPU
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // On essaie un CPU si pas de GPU
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        CHECK_CL(err, "clGetDeviceIDs (CPU)");
    }

    // Contexte
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    // File de commandes
#if CL_TARGET_OPENCL_VERSION >= 200
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CHECK_CL(err, "clCreateCommandQueue");

    // Programme et compilation
    const char *sources[] = { kernelSource };
    size_t lengths[] = { (size_t)strlen(kernelSource) };

    cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Affichage du log de compilation en cas d'erreur
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Erreur de compilation du programme OpenCL :\n%s\n", log);
        free(log);
        CHECK_CL(err, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "floyd_step", &err);
    CHECK_CL(err, "clCreateKernel");

    // -----------------------------
    // 3. Buffer sur le device
    // -----------------------------
    cl_mem d_dist = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   n * n * sizeof(int),
                                   dist,
                                   &err);
    CHECK_CL(err, "clCreateBuffer(dist)");

    // -----------------------------
    // 4. Boucle de Floyd–Warshall
    // -----------------------------
    size_t globalSize[2] = { (size_t)n, (size_t)n };

    for (int k = 0; k < n; ++k) {
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_dist);
        err |= clSetKernelArg(kernel, 1, sizeof(int),    &k);
        err |= clSetKernelArg(kernel, 2, sizeof(int),    &n);
        CHECK_CL(err, "clSetKernelArg");

        err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     2,           // dimensions (i, j)
                                     NULL,
                                     globalSize,  // taille globale
                                     NULL,        // taille locale : laissée à l'implémentation
                                     0,
                                     NULL,
                                     NULL);
        CHECK_CL(err, "clEnqueueNDRangeKernel");

        // On attend la fin de cette itération de k
        err = clFinish(queue);
        CHECK_CL(err, "clFinish");
    }

    // -----------------------------
    // 5. Récupération du résultat
    // -----------------------------
    err = clEnqueueReadBuffer(queue,
                              d_dist,
                              CL_TRUE,
                              0,
                              n * n * sizeof(int),
                              dist,
                              0,
                              NULL,
                              NULL);
    CHECK_CL(err, "clEnqueueReadBuffer");

    // -----------------------------
    // 6. (Optionnel) Affichage des distances
    // -----------------------------
    /*
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", dist[i * n + j]);
        }
        printf("\n");
    }
    */

    // -----------------------------
    // 7. Libération des ressources
    // -----------------------------
    clReleaseMemObject(d_dist);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(dist);

    return EXIT_SUCCESS;
}
