#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <iomanip>
#include <cmath>
#include <set>
#include <queue>
#include <fstream>
#include <sstream>

// TBB for parallelization
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

// CLIPPER
#include <clipper/clipper.h>
#include <clipper/utils.h>

// Local includes
#include "downsampling.hpp"
#include "FasterPFH.hpp"

// Forward declarations
clipper::Association get_a2a_assoc_matrix(int N1, int N2);
clipper::Association get_downsampled_a2a_assoc_matrix(int N1, int N2, int max_associations);
std::tuple<clipper::Association, std::vector<int>> downsampleAssociationMatrix(
    const clipper::Association& A,
    const std::vector<int>& corr_labels,
    int max_associations);
Eigen::Matrix4d umeyamaAlignment(const Eigen::MatrixXd& target, const Eigen::MatrixXd& source);
Eigen::Affine3d computeTransformationFromInliers(
    const Eigen::MatrixXd& target_cloud,
    const Eigen::MatrixXd& source_cloud,
    const clipper::Association& corres);
std::tuple<Eigen::Affine3d, clipper::Association> mergeMaps(
    const Eigen::MatrixXd& target_cloud_orig,
    const Eigen::MatrixXd& tf_target_cloud,
    const Eigen::MatrixXd& source_cloud_orig,
    const Eigen::MatrixXd& tf_source_cloud,
    const clipper::Association& associations,
    double map_voxel_size);

// Convert PCL point cloud to Eigen matrix (XYZ only)
Eigen::MatrixXd PCLToEigenXYZ(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) {
    Eigen::MatrixXd eigen_cloud(pcl_cloud->size(), 3);
    for (size_t i = 0; i < pcl_cloud->size(); ++i) {
        eigen_cloud(i, 0) = pcl_cloud->points[i].x;
        eigen_cloud(i, 1) = pcl_cloud->points[i].y;
        eigen_cloud(i, 2) = pcl_cloud->points[i].z;
    }
    return eigen_cloud;
}

// Convert Eigen::Vector3f vector to Eigen::MatrixXd
Eigen::MatrixXd vectorToEigenMatrix(const std::vector<Eigen::Vector3f>& points) {
    Eigen::MatrixXd matrix(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i) {
        matrix(i, 0) = points[i](0);
        matrix(i, 1) = points[i](1);
        matrix(i, 2) = points[i](2);
    }
    return matrix;
}

// Randomly sample up to max_points from a point cloud
std::vector<Eigen::Vector3f> randomSamplePoints(const std::vector<Eigen::Vector3f>& points, int max_points) {
    if (points.size() <= static_cast<size_t>(max_points)) {
        return points;  // Return all points if we have fewer than max_points
    }
    
    // Generate random indices
    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Sample the first max_points indices
    std::vector<Eigen::Vector3f> sampled;
    sampled.reserve(max_points);
    for (int i = 0; i < max_points; ++i) {
        sampled.push_back(points[indices[i]]);
    }
    
    return sampled;
}

// Randomly sample points and return both sampled points and their original indices
std::tuple<std::vector<Eigen::Vector3f>, std::vector<size_t>> randomSamplePointsWithIndices(
    const std::vector<Eigen::Vector3f>& points, int max_points) {
    std::vector<size_t> original_indices;
    
    if (points.size() <= static_cast<size_t>(max_points)) {
        // Return all points with their indices
        original_indices.resize(points.size());
        std::iota(original_indices.begin(), original_indices.end(), 0);
        return std::make_tuple(points, original_indices);
    }
    
    // Generate random indices
    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Sample the first max_points indices
    std::vector<Eigen::Vector3f> sampled;
    sampled.reserve(max_points);
    original_indices.reserve(max_points);
    
    for (int i = 0; i < max_points; ++i) {
        sampled.push_back(points[indices[i]]);
        original_indices.push_back(indices[i]);
    }
    
    return std::make_tuple(sampled, original_indices);
}

// Compute histogram similarity (L2 distance - smaller is more similar)
float computeHistogramSimilarity(const Eigen::VectorXf& hist1, const Eigen::VectorXf& hist2) {
    return (hist1 - hist2).norm();
}

// Filter FPFH descriptors to keep only those between specified percentile range
// Returns filtered descriptors and their corresponding point indices
std::tuple<std::vector<Eigen::VectorXf>, std::vector<Eigen::Vector3f>, std::vector<size_t>> 
filterDescriptorsByPercentile(
    const std::vector<Eigen::VectorXf>& descriptors,
    const std::vector<Eigen::Vector3f>& points,
    double percentile_lower,
    double percentile_upper)
{
    if (descriptors.empty()) {
        return std::make_tuple(std::vector<Eigen::VectorXf>(), 
                              std::vector<Eigen::Vector3f>(), 
                              std::vector<size_t>());
    }
    
    // Compute magnitudes for all descriptors
    std::vector<float> magnitudes;
    magnitudes.reserve(descriptors.size());
    for (const auto& desc : descriptors) {
        magnitudes.push_back(desc.norm());
    }
    
    // Sort magnitudes to find percentile thresholds
    std::vector<float> sorted_magnitudes = magnitudes;
    std::sort(sorted_magnitudes.begin(), sorted_magnitudes.end());
    
    // Find the lower and upper thresholds
    size_t lower_threshold_idx = static_cast<size_t>(sorted_magnitudes.size() * percentile_lower / 100.0);
    if (lower_threshold_idx >= sorted_magnitudes.size()) {
        lower_threshold_idx = sorted_magnitudes.size() - 1;
    }
    float lower_threshold = sorted_magnitudes[lower_threshold_idx];
    
    size_t upper_threshold_idx = static_cast<size_t>(sorted_magnitudes.size() * percentile_upper / 100.0);
    if (upper_threshold_idx >= sorted_magnitudes.size()) {
        upper_threshold_idx = sorted_magnitudes.size() - 1;
    }
    float upper_threshold = sorted_magnitudes[upper_threshold_idx];
    
    std::cout << "  Filtering descriptors: keeping between " << percentile_lower 
              << "th and " << percentile_upper << "th percentile" << std::endl;
    std::cout << "    Lower threshold: " << lower_threshold 
              << ", Upper threshold: " << upper_threshold << std::endl;
    
    // Filter descriptors and points - keep those between percentiles
    std::vector<Eigen::VectorXf> filtered_descriptors;
    std::vector<Eigen::Vector3f> filtered_points;
    std::vector<size_t> original_indices;
    
    // Reserve space for approximately (percentile_upper - percentile_lower)% of descriptors
    double percentile_range = percentile_upper - percentile_lower;
    filtered_descriptors.reserve(descriptors.size() * percentile_range / 100);
    filtered_points.reserve(points.size() * percentile_range / 100);
    original_indices.reserve(points.size() * percentile_range / 100);
    
    for (size_t i = 0; i < descriptors.size(); ++i) {
        // Keep if magnitude is between lower and upper thresholds
        if (magnitudes[i] >= lower_threshold && magnitudes[i] <= upper_threshold) {
            filtered_descriptors.push_back(descriptors[i]);
            if (i < points.size()) {
                filtered_points.push_back(points[i]);
            }
            original_indices.push_back(i);
        }
    }
    
    std::cout << "  Filtered from " << descriptors.size() << " to " 
              << filtered_descriptors.size() << " descriptors" << std::endl;
    
    return std::make_tuple(filtered_descriptors, filtered_points, original_indices);
}

// Build associations based on histogram similarity
// Memory-efficient version: uses priority queue to keep only top N similarities
clipper::Association buildHistogramBasedAssociations(
    const std::vector<Eigen::VectorXf>& target_histograms,
    const std::vector<Eigen::VectorXf>& source_histograms,
    int max_associations)
{
    std::cout << "Computing histogram similarities (parallelized, memory-efficient)..." << std::endl;
    std::cout << "  Target histograms: " << target_histograms.size() << std::endl;
    std::cout << "  Source histograms: " << source_histograms.size() << std::endl;
    std::cout << "  Max associations: " << max_associations << std::endl;
    
    // Use a max-heap (priority queue) to keep only the top max_associations smallest similarities
    // The heap stores (similarity, target_idx, source_idx) and we want the smallest similarities
    // So we use a max-heap with inverted comparison (largest similarity at top gets popped)
    using SimilarityTuple = std::tuple<float, int, int>;
    auto compare = [](const SimilarityTuple& a, const SimilarityTuple& b) {
        // Max-heap: larger similarity values have higher priority (get popped first)
        return std::get<0>(a) < std::get<0>(b);
    };
    
    // Thread-local heaps to avoid contention - each thread maintains its own top N
    using ThreadLocalHeap = std::priority_queue<SimilarityTuple, std::vector<SimilarityTuple>, decltype(compare)>;
    tbb::enumerable_thread_specific<ThreadLocalHeap> thread_local_heaps(compare);
    
    // Parallelize over target histograms
    tbb::parallel_for(tbb::blocked_range<size_t>(0, target_histograms.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            ThreadLocalHeap& local_heap = thread_local_heaps.local();
            
            for (size_t i = range.begin(); i < range.end(); ++i) {
                for (size_t j = 0; j < source_histograms.size(); ++j) {
                    float sim = computeHistogramSimilarity(target_histograms[i], source_histograms[j]);
                    
                    if (local_heap.size() < static_cast<size_t>(max_associations)) {
                        // Heap not full yet, just add
                        local_heap.emplace(sim, i, j);
                    } else {
                        // Heap is full, check if this similarity is better than worst in heap
                        if (sim < std::get<0>(local_heap.top())) {
                            local_heap.pop();
                            local_heap.emplace(sim, i, j);
                        }
                    }
                }
            }
        });
    
    // Merge all thread-local heaps into final heap
    std::priority_queue<SimilarityTuple, std::vector<SimilarityTuple>, decltype(compare)> top_similarities(compare);
    
    for (auto& local_heap : thread_local_heaps) {
        while (!local_heap.empty()) {
            const auto& item = local_heap.top();
            if (top_similarities.size() < static_cast<size_t>(max_associations)) {
                top_similarities.push(item);
            } else if (std::get<0>(item) < std::get<0>(top_similarities.top())) {
                top_similarities.pop();
                top_similarities.push(item);
            }
            local_heap.pop();
        }
    }
    
    std::cout << "Computed similarities and selected top " << top_similarities.size() << " pairs" << std::endl;
    
    // Convert heap to association matrix (heap is max-heap, so we need to reverse order)
    std::vector<SimilarityTuple> sorted_results;
    sorted_results.reserve(top_similarities.size());
    while (!top_similarities.empty()) {
        sorted_results.push_back(top_similarities.top());
        top_similarities.pop();
    }
    // Reverse to get ascending order (smallest first)
    std::reverse(sorted_results.begin(), sorted_results.end());
    
    int actual_max = sorted_results.size();
    clipper::Association assoc_matrix(actual_max, 2);
    
    for (int i = 0; i < actual_max; ++i) {
        int target_idx = std::get<1>(sorted_results[i]);
        int source_idx = std::get<2>(sorted_results[i]);
        assoc_matrix(i, 0) = target_idx;
        assoc_matrix(i, 1) = source_idx;
    }
    
    std::cout << "Selected top " << actual_max << " most similar associations" << std::endl;
    if (actual_max > 0) {
        std::cout << "  Best similarity: " << std::get<0>(sorted_results[0]) << std::endl;
        std::cout << "  Worst similarity: " << std::get<0>(sorted_results[actual_max - 1]) << std::endl;
    }
    
    return assoc_matrix;
}

std::tuple<Eigen::Affine3d, clipper::Association> mergeMaps(
    const Eigen::MatrixXd& /* target_cloud_orig */,
    const Eigen::MatrixXd& tf_target_cloud,
    const Eigen::MatrixXd& /* source_cloud_orig */,
    const Eigen::MatrixXd& tf_source_cloud,
    const clipper::Association& associations,
    double map_voxel_size) 
{
    // Set epsilon according to input map's voxel leaf len
    // a multiplier of 5 seemed to work well for many tests
    double eps = map_voxel_size * 5;

    // instantiate the invariant function that will be used to score associations
    clipper::invariants::EuclideanDistance::Params iparams;
    iparams.epsilon = eps;
    iparams.sigma = 0.5 * iparams.epsilon;
    clipper::invariants::EuclideanDistancePtr invariant =
              std::make_shared<clipper::invariants::EuclideanDistance>(iparams);

    // set up CLIPPER rounding parameters
    clipper::Params params;
    params.rounding = clipper::Params::Rounding::DSD_HEU;

    // instantiate clipper object
    clipper::CLIPPER clipper(invariant, params);
    
    Eigen::MatrixXd tf_target_cloud_points = tf_target_cloud.leftCols(3);
    Eigen::MatrixXd tf_source_cloud_points = tf_source_cloud.leftCols(3);

    // Score using invariant above and solve for maximal clique
    std::cout << "SCORING NOW" << std::endl;
    clipper.scorePairwiseConsistency(tf_target_cloud_points.transpose(), tf_source_cloud_points.transpose(), associations);

    std::cout << "SOLVING NOW" << std::endl;
    clipper.solveAsMaximumClique();

    // Retrieve selected inliers
    clipper::Association Ainliers = clipper.getSelectedAssociations();
    std::cout << "Ainliers_len: " << Ainliers.rows() << std::endl;

    // Compute peer2peer TF estimate
    std::cout << "COMPUTING TF" << std::endl;
    Eigen::Affine3d tf_est_affine = computeTransformationFromInliers(tf_target_cloud_points, tf_source_cloud_points, Ainliers);

    return std::make_tuple(tf_est_affine, Ainliers);
}

// All-to-All Association matrix (creates full matrix - use with caution for large point clouds)
clipper::Association get_a2a_assoc_matrix(int N1, int N2) 
{
    clipper::Association assoc_matrix(N1 * N2, 2);
    int i = 0;
    for (int n1 = 0; n1 < N1; ++n1) {
        for (int n2 = 0; n2 < N2; ++n2) {
            assoc_matrix(i, 0) = n1;
            assoc_matrix(i, 1) = n2;
            ++i;
        }
    }
    return assoc_matrix;
}

// Generate downsampled all-to-all associations directly (avoids creating huge matrix)
clipper::Association get_downsampled_a2a_assoc_matrix(int N1, int N2, int max_associations)
{
    long long total_associations = static_cast<long long>(N1) * static_cast<long long>(N2);
    int actual_max = std::min(static_cast<int>(total_associations), max_associations);
    
    if (actual_max >= total_associations) {
        // If we need all associations, use the full method
        return get_a2a_assoc_matrix(N1, N2);
    }
    
    // Generate random associations
    std::vector<std::pair<int, int>> associations;
    associations.reserve(actual_max);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist1(0, N1 - 1);
    std::uniform_int_distribution<int> dist2(0, N2 - 1);
    
    // Use a set to ensure uniqueness
    std::set<std::pair<int, int>> seen;
    
    while (associations.size() < static_cast<size_t>(actual_max)) {
        int n1 = dist1(gen);
        int n2 = dist2(gen);
        std::pair<int, int> pair(n1, n2);
        
        // Check for uniqueness
        if (seen.find(pair) == seen.end()) {
            associations.push_back(pair);
            seen.insert(pair);
        }
    }
    
    // Convert to Eigen matrix
    clipper::Association assoc_matrix(associations.size(), 2);
    for (size_t i = 0; i < associations.size(); ++i) {
        assoc_matrix(i, 0) = associations[i].first;
        assoc_matrix(i, 1) = associations[i].second;
    }
    
    return assoc_matrix;
}

// Filter based on set max # of associations
std::tuple<clipper::Association, std::vector<int>> downsampleAssociationMatrix(
    const clipper::Association& A,
    const std::vector<int>& corr_labels,
    int max_associations)
{
    int N = A.rows();
    int max_size_A = std::min(max_associations, N);  // avoid overflow

    // Generate random unique indices
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);  // fill with 0..N-1

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Downsample A
    std::vector<int> rand_ds_A_idxs(indices.begin(), indices.begin() + max_size_A);
    clipper::Association A_ds(max_size_A, 2);
    for (int i = 0; i < max_size_A; ++i) {
        A_ds.row(i) = A.row(rand_ds_A_idxs[i]);
    }

    // Downsample labels if provided
    std::vector<int> corr_labels_ds;
    if (!corr_labels.empty()) {
        corr_labels_ds.resize(max_size_A);
        for (int i = 0; i < max_size_A; ++i) {
            corr_labels_ds[i] = corr_labels[rand_ds_A_idxs[i]];
        }
    }

    return std::make_tuple(A_ds, corr_labels_ds);
}

Eigen::Matrix4d umeyamaAlignment(
    const Eigen::MatrixXd& target, 
    const Eigen::MatrixXd& source) 
{
    assert(target.rows() == source.rows());

    // compute centroids
    Eigen::Vector3d target_mean = target.colwise().mean();
    Eigen::Vector3d source_mean = source.colwise().mean();

    // center points
    Eigen::MatrixXd target_centered = target.rowwise() - target_mean.transpose();
    Eigen::MatrixXd source_centered = source.rowwise() - source_mean.transpose();

    // compute covariance matrix
    Eigen::Matrix3d H = target_centered.transpose() * source_centered;

    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // compute rotation
    Eigen::Matrix3d R = V * U.transpose();

    // ensure proper rotation (no reflection)
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // compute translation
    Eigen::Vector3d t = source_mean - R * target_mean;

    // construct transformation matrix
    Eigen::Matrix4d Tfmat = Eigen::Matrix4d::Identity();
    Tfmat.block<3,3>(0,0) = R;
    Tfmat.block<3,1>(0,3) = t;

    return Tfmat;
}

Eigen::Affine3d computeTransformationFromInliers(
    const Eigen::MatrixXd& target_cloud,
    const Eigen::MatrixXd& source_cloud,
    const clipper::Association& corres) 
{
    int N = corres.rows();
    Eigen::MatrixXd target_corr(N, 3);
    Eigen::MatrixXd source_corr(N, 3);

    for (int i = 0; i < N; ++i) {
        int target_idx = corres(i, 0);
        int source_idx = corres(i, 1);

        target_corr.row(i) = target_cloud.row(target_idx);
        source_corr.row(i) = source_cloud.row(source_idx);
    }

    // align dem bad boys
    Eigen::Matrix4d tf_est = umeyamaAlignment(source_corr, target_corr);
    Eigen::Affine3d tf_est_affine(tf_est);

    return tf_est_affine;
}

void printTransform(const Eigen::Affine3d& tf) {
    Eigen::Matrix4d matrix = tf.matrix();
    std::cout << "\n=== Estimated Transformation Matrix ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(12) << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n=== Rotation (Euler angles in degrees) ===" << std::endl;
    Eigen::Vector3d euler = tf.rotation().eulerAngles(0, 1, 2) * 180.0 / M_PI;
    std::cout << "Roll (X):  " << euler(0) << " degrees" << std::endl;
    std::cout << "Pitch (Y): " << euler(1) << " degrees" << std::endl;
    std::cout << "Yaw (Z):   " << euler(2) << " degrees" << std::endl;
    std::cout << "\n=== Translation ===" << std::endl;
    std::cout << "X: " << tf.translation()(0) << " m" << std::endl;
    std::cout << "Y: " << tf.translation()(1) << " m" << std::endl;
    std::cout << "Z: " << tf.translation()(2) << " m" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// Convert Eigen matrix to PCL point cloud with color
pcl::PointCloud<pcl::PointXYZRGB>::Ptr eigenToPCLColored(
    const Eigen::MatrixXd& eigen_cloud,
    uint8_t r, uint8_t g, uint8_t b)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl_cloud->points.reserve(eigen_cloud.rows());
    
    for (int i = 0; i < eigen_cloud.rows(); ++i) {
        pcl::PointXYZRGB pt;
        pt.x = eigen_cloud(i, 0);
        pt.y = eigen_cloud(i, 1);
        pt.z = eigen_cloud(i, 2);
        pt.r = r;
        pt.g = g;
        pt.b = b;
        pcl_cloud->points.push_back(pt);
    }
    
    pcl_cloud->width = pcl_cloud->points.size();
    pcl_cloud->height = 1;
    pcl_cloud->is_dense = false;
    
    return pcl_cloud;
}

// Transform Eigen matrix using Affine3d
Eigen::MatrixXd transformEigenMatrix(const Eigen::MatrixXd& cloud, const Eigen::Affine3d& tf) {
    Eigen::MatrixXd transformed(cloud.rows(), 3);
    
    // Get the transformation matrix
    Eigen::Matrix4d T = tf.matrix();
    
    // Extract rotation and translation
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    
    // Transform each point: p' = R * p + t
    for (int i = 0; i < cloud.rows(); ++i) {
        Eigen::Vector3d pt(cloud(i, 0), cloud(i, 1), cloud(i, 2));
        Eigen::Vector3d pt_tf = R * pt + t;
        transformed(i, 0) = pt_tf(0);
        transformed(i, 1) = pt_tf(1);
        transformed(i, 2) = pt_tf(2);
    }
    return transformed;
}

// Load transformation matrix from file
Eigen::Affine3d loadTransformationFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open transformation file " << filename << std::endl;
        return Eigen::Affine3d::Identity();
    }
    
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    
    for (int i = 0; i < 4; ++i) {
        std::string line;
        if (!std::getline(file, line)) {
            std::cerr << "Error: Transformation file has fewer than 4 rows" << std::endl;
            return Eigen::Affine3d::Identity();
        }
        
        std::istringstream iss(line);
        for (int j = 0; j < 4; ++j) {
            if (!(iss >> matrix(i, j))) {
                std::cerr << "Error: Could not parse transformation matrix at row " << i << ", col " << j << std::endl;
                return Eigen::Affine3d::Identity();
            }
        }
    }
    
    file.close();
    
    std::cout << "Loaded transformation matrix from " << filename << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(12) << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    // Print rotation and translation components for verification
    Eigen::Affine3d transform(matrix);
    std::cout << "\nRotation matrix:" << std::endl;
    Eigen::Matrix3d R = transform.rotation();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << std::setw(12) << R(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nTranslation vector: [" 
              << transform.translation()(0) << ", "
              << transform.translation()(1) << ", "
              << transform.translation()(2) << "]" << std::endl;
    std::cout << std::endl;
    
    return transform;
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool transform_mode = false;
    bool view_only = false;  // If true, just view without transformation
    bool transform_target = false;  // If true, transform target; if false, transform source
    bool use_inverse = false;  // If true, use inverse transformation
    std::string transform_file = "data/TRANSFORMATION.txt";
    std::string initial_transform_file = "data/TRANSFORMATION.txt";  // File for initial alignment
    std::string target_file = "data/test0.pcd";
    std::string source_file = "data/train0.pcd";
    
    // Parameters (declare before argument parsing so they can be modified)
    double leaf_size = 0.2;  // Downsampling leaf size in meters
    int max_associations = 1500;  // Maximum number of associations to keep
    double fpfh_percentile_lower = 80.0;  // Lower percentile for FPFH filtering (default: 80th)
    double fpfh_percentile_upper = 90.0;  // Upper percentile for FPFH filtering (default: 90th)
    
    // Simple argument parsing
    int file_arg_count = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--view" || arg == "-v") {
            view_only = true;
        } else if (arg == "--transform" || arg == "-t") {
            transform_mode = true;
        } else if (arg == "--transform-file" && i + 1 < argc) {
            transform_file = argv[++i];
            transform_mode = true;
        } else if (arg == "--transform-target") {
            transform_target = true;
            transform_mode = true;
        } else if (arg == "--transform-source") {
            transform_target = false;
            transform_mode = true;
        } else if (arg == "--inverse") {
            use_inverse = true;
        } else if (arg == "--initial-transform" && i + 1 < argc) {
            initial_transform_file = argv[++i];
        } else if (arg == "--fpfh-percentile" && i + 1 < argc) {
            // Parse as "lower,upper" or just a single value (which will be used as upper, with 80 as lower)
            std::string percentile_str = argv[++i];
            size_t comma_pos = percentile_str.find(',');
            if (comma_pos != std::string::npos) {
                fpfh_percentile_lower = std::stod(percentile_str.substr(0, comma_pos));
                fpfh_percentile_upper = std::stod(percentile_str.substr(comma_pos + 1));
            } else {
                fpfh_percentile_upper = std::stod(percentile_str);
                fpfh_percentile_lower = 80.0;  // Default lower
            }
            if (fpfh_percentile_lower < 0.0 || fpfh_percentile_lower > 100.0 ||
                fpfh_percentile_upper < 0.0 || fpfh_percentile_upper > 100.0 ||
                fpfh_percentile_lower >= fpfh_percentile_upper) {
                std::cerr << "Warning: fpfh-percentile must be valid range between 0 and 100, with lower < upper. Using default 80,90" << std::endl;
                fpfh_percentile_lower = 80.0;
                fpfh_percentile_upper = 90.0;
            }
        } else if (!arg.empty() && arg[0] != '-') {
            // Regular file argument (not a flag)
            if (file_arg_count == 0) {
                target_file = arg;
                file_arg_count++;
            } else if (file_arg_count == 1) {
                source_file = arg;
                file_arg_count++;
            }
        }
    }
    
    if (view_only) {
        std::cout << "=== View Only Mode ===" << std::endl;
        std::cout << "Target file: " << target_file << std::endl;
        std::cout << "Source file: " << source_file << std::endl;
        std::cout << std::endl;
    } else if (transform_mode) {
        std::cout << "=== Transform and View Mode ===" << std::endl;
        std::cout << "Target file: " << target_file << std::endl;
        std::cout << "Source file: " << source_file << std::endl;
        std::cout << "Transformation file: " << transform_file << std::endl;
        std::cout << "Transform " << (transform_target ? "target" : "source") << " cloud" << std::endl;
        if (use_inverse) {
            std::cout << "Using inverse transformation" << std::endl;
        }
        std::cout << std::endl;
    } else {
        std::cout << "=== Map Matching ===" << std::endl;
        std::cout << "Target file: " << target_file << std::endl;
        std::cout << "Source file: " << source_file << std::endl;
        std::cout << "Initial transformation file: " << initial_transform_file << std::endl;
        std::cout << "Leaf size: " << leaf_size << " m" << std::endl;
        std::cout << "Max associations: " << max_associations << std::endl;
        std::cout << "FPFH percentile filter: " << fpfh_percentile_lower << "th to " << fpfh_percentile_upper << "th percentile" << std::endl;
        std::cout << std::endl;
    }
    
    // 1. Load point clouds using PCL
    std::cout << "Loading point clouds..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_file, *target_cloud) == -1) {
        std::cerr << "Couldn't read file " << target_file << std::endl;
        return -1;
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(source_file, *source_cloud) == -1) {
        std::cerr << "Couldn't read file " << source_file << std::endl;
        return -1;
    }
    
    std::cout << "Loaded target cloud with " << target_cloud->points.size() << " points" << std::endl;
    std::cout << "Loaded source cloud with " << source_cloud->points.size() << " points" << std::endl;
    
    // If view-only mode, just visualize without transformation
    if (view_only) {
        // Convert to Eigen matrices
        Eigen::MatrixXd target_matrix = PCLToEigenXYZ(target_cloud);
        Eigen::MatrixXd source_matrix = PCLToEigenXYZ(source_cloud);
        
        // Convert to PCL point clouds with colors
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcl = eigenToPCLColored(target_matrix, 255, 0, 0);  // Red
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcl = eigenToPCLColored(source_matrix, 0, 0, 255);  // Blue
        
        // Visualize
        std::cout << "Visualizing point clouds (no transformation)..." << std::endl;
        std::cout << "  Target cloud: Red" << std::endl;
        std::cout << "  Source cloud: Blue" << std::endl;
        std::cout << "Press 'q' to quit the viewer" << std::endl;
        
        // Create visualizer
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Clouds (No Transformation)"));
        viewer->setBackgroundColor(1.0, 1.0, 1.0);  // White background
        
        // Add point clouds
        viewer->addPointCloud<pcl::PointXYZRGB>(target_pcl, "target_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_cloud");
        
        viewer->addPointCloud<pcl::PointXYZRGB>(source_pcl, "source_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source_cloud");
        
        // Add coordinate system
        viewer->addCoordinateSystem(10.0);
        viewer->initCameraParameters();
        
        // Run viewer
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
        }
        
        return 0;
    }
    
    // If transform mode, just transform and visualize
    if (transform_mode) {
        // Load transformation matrix
        Eigen::Affine3d transform = loadTransformationFromFile(transform_file);
        
        // Apply inverse if requested
        if (use_inverse) {
            std::cout << "Using inverse transformation..." << std::endl;
            transform = transform.inverse();
        }
        
        // Convert to Eigen matrices
        Eigen::MatrixXd target_matrix = PCLToEigenXYZ(target_cloud);
        Eigen::MatrixXd source_matrix = PCLToEigenXYZ(source_cloud);
        
        // Apply transformation based on which cloud to transform
        Eigen::MatrixXd target_transformed, source_transformed;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcl, source_pcl;
        
        if (transform_target) {
            // Transform target cloud
            std::cout << "Applying transformation to target cloud..." << std::endl;
            target_transformed = transformEigenMatrix(target_matrix, transform);
            source_transformed = source_matrix;
            
            target_pcl = eigenToPCLColored(target_transformed, 255, 0, 0);  // Red (transformed)
            source_pcl = eigenToPCLColored(source_transformed, 0, 0, 255);  // Blue (original)
            
            std::cout << "Visualizing point clouds..." << std::endl;
            std::cout << "  Transformed target cloud: Red" << std::endl;
            std::cout << "  Source cloud: Blue" << std::endl;
        } else {
            // Transform source cloud (default)
            std::cout << "Applying transformation to source cloud..." << std::endl;
            target_transformed = target_matrix;
            source_transformed = transformEigenMatrix(source_matrix, transform);
            
            target_pcl = eigenToPCLColored(target_transformed, 255, 0, 0);  // Red (original)
            source_pcl = eigenToPCLColored(source_transformed, 0, 0, 255);  // Blue (transformed)
            
            std::cout << "Visualizing point clouds..." << std::endl;
            std::cout << "  Target cloud: Red" << std::endl;
            std::cout << "  Transformed source cloud: Blue" << std::endl;
        }
        
        std::cout << "Press 'q' to quit the viewer" << std::endl;
        
        // Create visualizer
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Transformed Point Clouds"));
        viewer->setBackgroundColor(1.0, 1.0, 1.0);  // White background
        
        // Add point clouds
        viewer->addPointCloud<pcl::PointXYZRGB>(target_pcl, "target_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_cloud");
        
        viewer->addPointCloud<pcl::PointXYZRGB>(source_pcl, "source_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source_cloud");
        
        // Add coordinate system
        viewer->addCoordinateSystem(10.0);
        viewer->initCameraParameters();
        
        // Run viewer
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
        }
        
        return 0;
    }
    
    // 1.5. Apply initial transformation to source cloud if provided (for matching mode only)
    if (!view_only && !transform_mode) {
        std::cout << "\nApplying initial transformation to source cloud..." << std::endl;
        Eigen::Affine3d initial_transform = loadTransformationFromFile(initial_transform_file);
        
        // Convert source cloud to Eigen matrix
        Eigen::MatrixXd source_matrix = PCLToEigenXYZ(source_cloud);
        
        // Apply transformation
        Eigen::MatrixXd source_transformed = transformEigenMatrix(source_matrix, initial_transform);
        
        // Convert back to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        transformed_source_cloud->points.reserve(source_transformed.rows());
        for (int i = 0; i < source_transformed.rows(); ++i) {
            pcl::PointXYZ pt;
            pt.x = source_transformed(i, 0);
            pt.y = source_transformed(i, 1);
            pt.z = source_transformed(i, 2);
            transformed_source_cloud->points.push_back(pt);
        }
        transformed_source_cloud->width = transformed_source_cloud->points.size();
        transformed_source_cloud->height = 1;
        transformed_source_cloud->is_dense = false;
        
        // Replace source cloud with transformed version
        source_cloud = transformed_source_cloud;
        std::cout << "Applied initial transformation to " << source_cloud->points.size() << " source points" << std::endl;
    }
    
    // 2. Convert to Eigen::Vector3f for downsampling
    std::vector<Eigen::Vector3f> target_points;
    target_points.reserve(target_cloud->points.size());
    for (const auto& pt : target_cloud->points) {
        target_points.emplace_back(pt.x, pt.y, pt.z);
    }
    
    std::vector<Eigen::Vector3f> source_points;
    source_points.reserve(source_cloud->points.size());
    for (const auto& pt : source_cloud->points) {
        source_points.emplace_back(pt.x, pt.y, pt.z);
    }
    
    // 3. Downsample each map
    std::cout << "\nDownsampling point clouds..." << std::endl;
    std::vector<Eigen::Vector3f> target_downsampled = kiss_matcher::VoxelgridSampling(target_points, leaf_size);
    std::vector<Eigen::Vector3f> source_downsampled = kiss_matcher::VoxelgridSampling(source_points, leaf_size);
    
    std::cout << "Target: " << target_points.size() << " -> " << target_downsampled.size() << " points" << std::endl;
    std::cout << "Source: " << source_points.size() << " -> " << source_downsampled.size() << " points" << std::endl;
    
    // 4. Compute FPFH histograms for downsampled clouds
    std::cout << "\nComputing FPFH histograms..." << std::endl;
    float normal_radius = leaf_size * 2.0f;
    float fpfh_radius = leaf_size * 4.0f;
    float thr_linearity = 0.5f;
    
    // Compute histograms for target

    kiss_matcher::FasterPFH fpfh_target(normal_radius, fpfh_radius, thr_linearity, "L2", true);
    fpfh_target.setInputCloud(target_downsampled);
    std::vector<Eigen::Vector3f> target_valid_points;
    std::vector<Eigen::VectorXf> target_descriptors;
    fpfh_target.ComputeFeature(target_valid_points, target_descriptors);
    std::cout << "Target: computed " << target_descriptors.size() << " FPFH descriptors from " 
              << target_downsampled.size() << " downsampled points" << std::endl;
    
    // Compute histograms for source
    kiss_matcher::FasterPFH fpfh_source(normal_radius, fpfh_radius, thr_linearity, "L2", true);
    fpfh_source.setInputCloud(source_downsampled);
    std::vector<Eigen::Vector3f> source_valid_points;
    std::vector<Eigen::VectorXf> source_descriptors;
    fpfh_source.ComputeFeature(source_valid_points, source_descriptors);
    std::cout << "Source: computed " << source_descriptors.size() << " FPFH descriptors from " 
              << source_downsampled.size() << " downsampled points" << std::endl;
    
    // Check if we have valid descriptors
    if (target_descriptors.empty() || source_descriptors.empty()) {
        std::cerr << "Error: No FPFH descriptors computed. Check point cloud density and radius parameters." << std::endl;
        return -1;
    }
    
    // 5. Filter descriptors to keep only those between specified percentile range
    std::cout << "\nFiltering FPFH descriptors by percentile..." << std::endl;
    std::cout << "Target descriptors:" << std::endl;
    auto [target_filtered_descriptors, target_filtered_points, target_original_indices] = 
        filterDescriptorsByPercentile(target_descriptors, target_valid_points, fpfh_percentile_lower, fpfh_percentile_upper);
    
    std::cout << "Source descriptors:" << std::endl;
    auto [source_filtered_descriptors, source_filtered_points, source_original_indices] = 
        filterDescriptorsByPercentile(source_descriptors, source_valid_points, fpfh_percentile_lower, fpfh_percentile_upper);
    
    // Check if we have enough filtered descriptors
    if (target_filtered_descriptors.empty() || source_filtered_descriptors.empty()) {
        std::cerr << "Error: No descriptors remaining after filtering. Try a higher percentile threshold." << std::endl;
        return -1;
    }
    
    // 6. Build associations based on histogram similarity (using filtered descriptors)
    std::cout << "\nBuilding associations based on histogram similarity..." << std::endl;
    std::cout << "  Target filtered descriptors: " << target_filtered_descriptors.size() << std::endl;
    std::cout << "  Source filtered descriptors: " << source_filtered_descriptors.size() << std::endl;
    clipper::Association associations = buildHistogramBasedAssociations(
        target_filtered_descriptors, source_filtered_descriptors, max_associations);
    
    // Map association indices back to original point indices
    // The associations use indices into filtered_descriptors, but we need indices into valid_points
    std::cout << "Mapping association indices to original point indices..." << std::endl;
    for (int i = 0; i < associations.rows(); ++i) {
        int filtered_target_idx = static_cast<int>(associations(i, 0));
        int filtered_source_idx = static_cast<int>(associations(i, 1));
        
        if (filtered_target_idx < static_cast<int>(target_original_indices.size()) &&
            filtered_source_idx < static_cast<int>(source_original_indices.size())) {
            associations(i, 0) = target_original_indices[filtered_target_idx];
            associations(i, 1) = source_original_indices[filtered_source_idx];
        }
    }
    
    // Convert valid points to Eigen matrices (use all points with valid descriptors for transformation)
    Eigen::MatrixXd target_cloud_orig = vectorToEigenMatrix(target_valid_points);
    Eigen::MatrixXd source_cloud_orig = vectorToEigenMatrix(source_valid_points);
    
    // For initial alignment, use identity transform (or you could use ICP for rough alignment)
    Eigen::MatrixXd tf_target_cloud = target_cloud_orig;
    Eigen::MatrixXd tf_source_cloud = source_cloud_orig;
    
    // 7. Estimate transformation using mergeMaps
    std::cout << "\nEstimating transformation..." << std::endl;
    auto [tf_est, selected_inliers] = mergeMaps(
        target_cloud_orig,
        tf_target_cloud,
        source_cloud_orig,
        tf_source_cloud,
        associations,
        leaf_size);
    
    std::cout << "Selected " << selected_inliers.rows() << " inlier associations from " 
              << associations.rows() << " total associations" << std::endl;
    
    // 8. Visualize point clouds BEFORE transformation with selected associations
    std::cout << "\nVisualizing point clouds BEFORE transformation..." << std::endl;
    std::cout << "  Target cloud: Red" << std::endl;
    std::cout << "  Source cloud: Blue" << std::endl;
    std::cout << "  Selected inlier associations: Green" << std::endl;
    std::cout << "Press 'q' to continue to transformed view" << std::endl;
    
    // Convert to PCL point clouds with colors (before transformation)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcl_before = eigenToPCLColored(target_cloud_orig, 255, 0, 0);  // Red
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcl_before = eigenToPCLColored(source_cloud_orig, 0, 0, 255);  // Blue
    
    // Create visualizer for before transformation
    pcl::visualization::PCLVisualizer::Ptr viewer_before(new pcl::visualization::PCLVisualizer("Point Clouds BEFORE Transformation"));
    viewer_before->setBackgroundColor(1.0, 1.0, 1.0);  // White background
    
    // Add point clouds
    viewer_before->addPointCloud<pcl::PointXYZRGB>(target_pcl_before, "target_cloud_before");
    viewer_before->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_cloud_before");
    
    viewer_before->addPointCloud<pcl::PointXYZRGB>(source_pcl_before, "source_cloud_before");
    viewer_before->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source_cloud_before");
    
    // Draw lines between matched points (selected inliers) - BEFORE transformation
    std::cout << "Drawing " << selected_inliers.rows() << " inlier association lines (before transformation)..." << std::endl;
    for (int i = 0; i < selected_inliers.rows(); ++i) {
        int target_idx = static_cast<int>(selected_inliers(i, 0));
        int source_idx = static_cast<int>(selected_inliers(i, 1));
        
        if (target_idx >= 0 && target_idx < target_cloud_orig.rows() &&
            source_idx >= 0 && source_idx < source_cloud_orig.rows()) {
            // Get points from original (untransformed) clouds
            pcl::PointXYZ pt1(target_cloud_orig(target_idx, 0), 
                             target_cloud_orig(target_idx, 1), 
                             target_cloud_orig(target_idx, 2));
            pcl::PointXYZ pt2(source_cloud_orig(source_idx, 0), 
                             source_cloud_orig(source_idx, 1), 
                             source_cloud_orig(source_idx, 2));
            
            // Create line ID
            std::string line_id = "inlier_line_before_" + std::to_string(i);
            
            // Add line (green color for inlier associations)
            viewer_before->addLine(pt1, pt2, 0.0, 1.0, 0.0, line_id);
        }
    }
    
    // Add coordinate system
    viewer_before->addCoordinateSystem(10.0);
    viewer_before->initCameraParameters();
    
    // Run viewer for before transformation
    while (!viewer_before->wasStopped()) {
        viewer_before->spinOnce(100);
    }
    
    // Close the before viewer
    viewer_before->close();
    
    // 9. Print the estimated transform
    printTransform(tf_est);
    
    // 10. Transform the source cloud using the estimated transformation
    std::cout << "Transforming source cloud..." << std::endl;
    Eigen::MatrixXd source_transformed = transformEigenMatrix(source_cloud_orig, tf_est);
    
    // 11. Visualize the point clouds AFTER transformation
    std::cout << "\nVisualizing point clouds AFTER transformation..." << std::endl;
    std::cout << "  Target cloud: Red" << std::endl;
    std::cout << "  Transformed source cloud: Blue" << std::endl;
    std::cout << "  Selected inlier associations: Green" << std::endl;
    std::cout << "Press 'q' to quit the viewer" << std::endl;
    
    // Convert to PCL point clouds with colors
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcl = eigenToPCLColored(target_cloud_orig, 255, 0, 0);  // Red
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcl = eigenToPCLColored(source_transformed, 0, 0, 255);  // Blue
    
    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Map Matching Result"));
    viewer->setBackgroundColor(1.0, 1.0, 1.0);  // White background
    
    // Add point clouds
    viewer->addPointCloud<pcl::PointXYZRGB>(target_pcl, "target_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_cloud");
    
    viewer->addPointCloud<pcl::PointXYZRGB>(source_pcl, "source_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source_cloud");
    
    // Draw lines between matched points (selected inliers only)
    std::cout << "Drawing " << selected_inliers.rows() << " inlier association lines..." << std::endl;
    for (int i = 0; i < selected_inliers.rows(); ++i) {
        int target_idx = static_cast<int>(selected_inliers(i, 0));
        int source_idx = static_cast<int>(selected_inliers(i, 1));
        
        if (target_idx >= 0 && target_idx < target_cloud_orig.rows() &&
            source_idx >= 0 && source_idx < source_transformed.rows()) {
            // Get points from target cloud (red) to transformed source cloud (blue)
            pcl::PointXYZ pt1(target_cloud_orig(target_idx, 0), 
                             target_cloud_orig(target_idx, 1), 
                             target_cloud_orig(target_idx, 2));
            pcl::PointXYZ pt2(source_transformed(source_idx, 0), 
                             source_transformed(source_idx, 1), 
                             source_transformed(source_idx, 2));
            
            // Create line ID
            std::string line_id = "inlier_line_" + std::to_string(i);
            
            // Add line (green color for inlier associations)
            viewer->addLine(pt1, pt2, 0.0, 1.0, 0.0, line_id);
        }
    }
    
    // Add coordinate system
    viewer->addCoordinateSystem(10.0);
    viewer->initCameraParameters();
    
    // Run viewer
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
    
    return 0;
}

