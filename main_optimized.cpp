#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include "KMeansOptimized.h"

using namespace std;

// #ifndef NUM_POINTS
// #define NUM_POINTS 500000
// #endif
// #ifndef NUM_CLUSTERS
// #define NUM_CLUSTERS 20
// #endif
// #ifndef MAX_ITERATIONS
// #define MAX_ITERATIONS 20
// #endif

double max_range = 100000;
int num_point = NUM_POINTS;
int num_cluster = NUM_CLUSTERS;
int max_iterations = MAX_ITERATIONS;

void draw_chart_gnu(const PointsSoA &points) {
    ofstream outfile("data_opt.txt");
    for (size_t i = 0; i < points.x.size(); i++) {
        outfile << points.x[i] << " " << points.y[i] << " " << points.cluster_id[i] << std::endl;
    }
    outfile.close();
    if (system("gnuplot -p -e \"plot 'data_opt.txt' using 1:2:3 with points palette notitle\"") < 0) {
        printf("Error: gnuplot could not be executed\n");
    }
    remove("data_opt.txt");
}

int main() {
    printf("--- Optimized KMeans (SoA + Local Accumulation) ---\n");
    printf("Number of points %d\n", num_point);
    printf("Number of clusters %d\n", num_cluster);
    printf("Number of processors: %d\n", omp_get_num_procs());

    srand(int(time(NULL)));

    double time_point1 = omp_get_wtime();
    printf("Starting initialization..\n");

    PointsSoA points(num_point);
    ClustersSoA clusters(num_cluster);

#pragma omp parallel
    {
        std::mt19937 gen(time(NULL) ^ omp_get_thread_num());
        std::uniform_real_distribution<> dis(0, max_range);

#pragma omp for schedule(static)
        for (int i = 0; i < num_point; i++) {
            points.x[i] = dis(gen);
            points.y[i] = dis(gen);
        }

#pragma omp for schedule(static)
        for (int i = 0; i < num_cluster; i++) {
            clusters.x[i] = dis(gen);
            clusters.y[i] = dis(gen);
        }
    }

    int num_threads = omp_get_max_threads();
    std::vector<PartialSums> thread_partials(num_threads, PartialSums(num_cluster));

    double time_point2 = omp_get_wtime();
    printf("Points and clusters generated in: %f seconds\n", time_point2 - time_point1);

    bool conv = true;
    int iterations = 0;

    printf("Starting iterate...\n");

    while (conv && iterations < max_iterations) {
        iterations++;
        compute_distance_optimized(points, clusters, thread_partials);
        conv = update_clusters_optimized(clusters, thread_partials);
        printf("Iteration %d done \n", iterations);
    }

    double time_point3 = omp_get_wtime();
    double duration = time_point3 - time_point2;

    printf("Number of iterations: %d, total time: %f seconds, time per iteration: %f seconds\n",
           iterations, duration, duration / iterations);

    try {
        printf("Drawing the chart...\n");
        // draw_chart_gnu(points); // Uncomment if gnuplot is available
    } catch (...) {
        printf("Chart not available\n");
    }

    return 0;
}
