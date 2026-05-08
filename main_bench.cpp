#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include "KMeansOptimized.h"

using namespace std;

#ifndef NUM_POINTS
#define NUM_POINTS 1000000
#endif
#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 50
#endif
#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS 10
#endif

double max_range = 100000;
int num_point = NUM_POINTS;
int num_cluster = NUM_CLUSTERS;
int max_iterations = MAX_ITERATIONS;

int main() {
    printf("--- KMeans Scheduling Benchmark ---\n");
    printf("Points: %d, Clusters: %d\n", num_point, num_cluster);

    PointsSoA points(num_point);
    ClustersSoA clusters(num_cluster);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, max_range);

    for (int i = 0; i < num_point; i++) {
        points.x[i] = dis(gen);
        points.y[i] = dis(gen);
    }
    for (int i = 0; i < num_cluster; i++) {
        clusters.x[i] = dis(gen);
        clusters.y[i] = dis(gen);
    }

    int num_threads = omp_get_max_threads();
    std::vector<PartialSums> thread_partials(num_threads, PartialSums(num_cluster));

    auto run_bench = [&](string name) {
        printf("Benchmarking %s...\n", name.c_str());
        auto start = chrono::high_resolution_clock::now();
        
        for (int i = 0; i < max_iterations; i++) {
            compute_distance_optimized(points, clusters, thread_partials);
            update_clusters_optimized(clusters, thread_partials);
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end - start;
        printf("%s: %f seconds\n", name.c_str(), diff.count());
    };

    // To test different schedules, you can use omp_set_schedule or just run with OMP_SCHEDULE env var
    // But since I used schedule(runtime), I can control it via environment.
    
    // In this script, I'll just run one and tell the user how to swap.
    run_bench("Current Schedule (via OMP_SCHEDULE)");

    return 0;
}
