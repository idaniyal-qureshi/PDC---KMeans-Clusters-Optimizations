__kernel void compute_distances_kernel(
    __global const double* points_x,
    __global const double* points_y,
    __global const double* clusters_x,
    __global const double* clusters_y,
    __global int* cluster_ids,
    const int num_points,
    const int num_clusters)
{
    int i = get_global_id(0);
    if (i >= num_points) return;

    double px = points_x[i];
    double py = points_y[i];

    double min_dist_sq = -1.0;
    int min_idx = 0;

    for (int j = 0; j < num_clusters; j++) {
        double dx = px - clusters_x[j];
        double dy = py - clusters_y[j];
        double dist_sq = dx * dx + dy * dy;

        if (min_dist_sq < 0 || dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            min_idx = j;
        }
    }

    cluster_ids[i] = min_idx;
}

// Fixed-point factor for atomic accumulation (to handle doubles)
#define FIXED_POINT_FACTOR 1000000LL

__kernel void clear_centroids_kernel(
    __global long* sum_x,
    __global long* sum_y,
    __global int* counts,
    const int num_clusters)
{
    int j = get_global_id(0);
    if (j >= num_clusters) return;
    sum_x[j] = 0;
    sum_y[j] = 0;
    counts[j] = 0;
}

__kernel void accumulate_centroids_kernel(
    __global const double* points_x,
    __global const double* points_y,
    __global const int* cluster_ids,
    __global volatile long* sum_x,
    __global volatile long* sum_y,
    __global volatile int* counts,
    const int num_points)
{
    int i = get_global_id(0);
    if (i >= num_points) return;

    int id = cluster_ids[i];
    
    // Atomically add to global sums using fixed-point
    atomic_add(&sum_x[id], (long)(points_x[i] * FIXED_POINT_FACTOR));
    atomic_add(&sum_y[id], (long)(points_y[i] * FIXED_POINT_FACTOR));
    atomic_add(&counts[id], 1);
}

__kernel void finalize_centroids_kernel(
    __global double* clusters_x,
    __global double* clusters_y,
    __global const long* sum_x,
    __global const long* sum_y,
    __global const int* counts,
    const int num_clusters)
{
    int j = get_global_id(0);
    if (j >= num_clusters) return;

    int count = counts[j];
    if (count > 0) {
        clusters_x[j] = (double)sum_x[j] / (double)(FIXED_POINT_FACTOR * (long)count);
        clusters_y[j] = (double)sum_y[j] / (double)(FIXED_POINT_FACTOR * (long)count);
    }
}