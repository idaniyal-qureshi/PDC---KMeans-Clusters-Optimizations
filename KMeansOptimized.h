#ifndef K_MEANS_OPTIMIZED_H
#define K_MEANS_OPTIMIZED_H

#include <vector>
#include <cmath>
#include <omp.h>

struct PointsSoA {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<int> cluster_id;

    PointsSoA(int n) : x(n), y(n), cluster_id(n, 0) {}
};

struct ClustersSoA {
    std::vector<double> x;
    std::vector<double> y;
    int size;

    ClustersSoA(int k) : x(k), y(k), size(k) {}
};

// Structure to hold partial sums for reduction
struct PartialSums {
    std::vector<double> sum_x;
    std::vector<double> sum_y;
    std::vector<int> count;

    PartialSums(int k) : sum_x(k, 0.0), sum_y(k, 0.0), count(k, 0) {}
};

inline double euclidean_dist_sq(double px, double py, double cx, double cy) {
    double dx = px - cx;
    double dy = py - cy;
    return dx * dx + dy * dy;
}

void compute_distance_optimized(PointsSoA &points, ClustersSoA &clusters, std::vector<PartialSums> &thread_partials) {
    int n = points.x.size();
    int k = clusters.x.size();
    int num_threads = omp_get_max_threads();

    // Reset partial sums
#pragma omp parallel for
    for (int t = 0; t < num_threads; t++) {
        std::fill(thread_partials[t].sum_x.begin(), thread_partials[t].sum_x.end(), 0.0);
        std::fill(thread_partials[t].sum_y.begin(), thread_partials[t].sum_y.end(), 0.0);
        std::fill(thread_partials[t].count.begin(), thread_partials[t].count.end(), 0);
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        PartialSums &my_sums = thread_partials[tid];

#pragma omp for schedule(runtime)
        for (int i = 0; i < n; i++) {
            double px = points.x[i];
            double py = points.y[i];

            double min_dist_sq = euclidean_dist_sq(px, py, clusters.x[0], clusters.y[0]);
            int min_idx = 0;

#pragma omp simd
            for (int j = 1; j < k; j++) {
                double dist_sq = euclidean_dist_sq(px, py, clusters.x[j], clusters.y[j]);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_idx = j;
                }
            }

            points.cluster_id[i] = min_idx;
            my_sums.sum_x[min_idx] += px;
            my_sums.sum_y[min_idx] += py;
            my_sums.count[min_idx]++;
        }
    }
}

bool update_clusters_optimized(ClustersSoA &clusters, const std::vector<PartialSums> &thread_partials) {
    int k = clusters.x.size();
    int num_threads = thread_partials.size();
    bool shifted = false;

    for (int j = 0; j < k; j++) {
        double total_x = 0.0;
        double total_y = 0.0;
        int total_count = 0;

        for (int t = 0; t < num_threads; t++) {
            total_x += thread_partials[t].sum_x[j];
            total_y += thread_partials[t].sum_y[j];
            total_count += thread_partials[t].count[j];
        }

        if (total_count > 0) {
            double new_x = total_x / total_count;
            double new_y = total_y / total_count;

            if (std::abs(clusters.x[j] - new_x) > 1e-6 || std::abs(clusters.y[j] - new_y) > 1e-6) {
                clusters.x[j] = new_x;
                clusters.y[j] = new_y;
                shifted = true;
            }
        }
    }

    return shifted;
}

#endif
