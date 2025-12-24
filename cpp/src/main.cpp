#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>
#include "FasterPFH.hpp"
#include "downsampling.hpp"

int main(int argc, char** argv) {
    // Load point cloud from PCD file
    std::string pcd_file;
    if (argc > 1) {
        pcd_file = argv[1];
    } else {
        // Default path - adjust based on where executable is run from
        pcd_file = "cpp/data/kitti00.pcd";
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1) {
        std::cerr << "Couldn't read file " << pcd_file << std::endl;
        return -1;
    }

    std::cout << "Loaded point cloud with " << cloud->points.size() << " points" << std::endl;

    // Convert PCL point cloud to Eigen::Vector3f format
    std::vector<Eigen::Vector3f> points;
    points.reserve(cloud->points.size());
    
    for (const auto& pt : cloud->points) {
        points.emplace_back(pt.x, pt.y, pt.z);
    }

    // Downsample point cloud using voxel grid with 2m leaf size
    double leaf_size = 1.0;  // 2 meters
    std::cout << "Downsampling point cloud with " << leaf_size << "m leaf size..." << std::endl;
    std::vector<Eigen::Vector3f> downsampled_points = kiss_matcher::VoxelgridSampling(points, leaf_size);
    std::cout << "Downsampled from " << points.size() << " to " << downsampled_points.size() << " points" << std::endl;

    // Use downsampled points for processing
    points = std::move(downsampled_points);

    // Initialize FasterPFH with appropriate parameters
    // KITTI data is typically in meters, so we use larger radii
    float normal_radius = leaf_size * 2.0f; // Radius for normal estimation (meters)
    float fpfh_radius = leaf_size * 5.0f;   // Radius for FPFH feature computation (meters)
    float thr_linearity = 0.0f;             // Linearity threshold for filtering

    kiss_matcher::FasterPFH fpfh(normal_radius, fpfh_radius, thr_linearity, "L2", false);
    
    // Set input cloud
    fpfh.setInputCloud(points);
    
    // Compute FPFH features
    std::cout << "Computing FPFH features..." << std::endl;
    std::vector<Eigen::Vector3f> valid_points;
    std::vector<Eigen::VectorXf> descriptors;
    
    fpfh.ComputeFeature(valid_points, descriptors);
    
    std::cout << "Computed FPFH descriptors for " << descriptors.size() << " points" << std::endl;

    // Calculate histogram magnitudes (L2 norm of each descriptor)
    std::vector<float> magnitudes;
    magnitudes.reserve(descriptors.size());
    
    for (const auto& desc : descriptors) {
        float magnitude = desc.norm();
        magnitudes.push_back(magnitude);
    }

    // Calculate percentile thresholds at 5% intervals (5, 10, 15, ..., 95)
    std::vector<float> sorted_magnitudes = magnitudes;
    std::sort(sorted_magnitudes.begin(), sorted_magnitudes.end());
    
    std::vector<float> percentile_thresholds;
    std::vector<double> percentiles;
    for (int p = 5; p <= 95; p += 5) {
        percentiles.push_back(p / 100.0);
    }
    
    for (double p : percentiles) {
        size_t idx = static_cast<size_t>(sorted_magnitudes.size() * p);
        if (idx >= sorted_magnitudes.size()) idx = sorted_magnitudes.size() - 1;
        percentile_thresholds.push_back(sorted_magnitudes[idx]);
    }
    
    std::cout << "Percentile thresholds (5% intervals):" << std::endl;
    for (size_t i = 0; i < percentiles.size(); ++i) {
        if (i % 4 == 0 || i == percentiles.size() - 1) {  // Print every 4th or last
            std::cout << "  " << (percentiles[i] * 100) << "th: " << percentile_thresholds[i] << std::endl;
        }
    }
    std::cout << "Min magnitude: " << sorted_magnitudes.front() << std::endl;
    std::cout << "Max magnitude: " << sorted_magnitudes.back() << std::endl;

    // Create colored point cloud based on percentile ranges
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->points.reserve(valid_points.size());
    
    // Generate smooth color gradient from blue (low) to red (high) through spectrum
    // Interpolate colors smoothly across 19 intervals (5% each)
    auto interpolate_color = [](double t, uint8_t& r, uint8_t& g, uint8_t& b) {
        // t is between 0.0 (low) and 1.0 (high)
        if (t < 0.25) {
            // Blue to Cyan: (0,0,255) -> (0,255,255)
            double local_t = t / 0.25;
            r = 0;
            g = static_cast<uint8_t>(255 * local_t);
            b = 255;
        } else if (t < 0.5) {
            // Cyan to Green: (0,255,255) -> (0,255,0)
            double local_t = (t - 0.25) / 0.25;
            r = 0;
            g = 255;
            b = static_cast<uint8_t>(255 * (1.0 - local_t));
        } else if (t < 0.75) {
            // Green to Yellow: (0,255,0) -> (255,255,0)
            double local_t = (t - 0.5) / 0.25;
            r = static_cast<uint8_t>(255 * local_t);
            g = 255;
            b = 0;
        } else {
            // Yellow to Red: (255,255,0) -> (255,0,0)
            double local_t = (t - 0.75) / 0.25;
            r = 255;
            g = static_cast<uint8_t>(255 * (1.0 - local_t));
            b = 0;
        }
    };
    
    for (size_t i = 0; i < valid_points.size(); ++i) {
        const auto& pt = valid_points[i];
        pcl::PointXYZRGB pcl_pt;
        pcl_pt.x = pt(0);
        pcl_pt.y = pt(1);
        pcl_pt.z = pt(2);
        
        // Determine which percentile range this point falls into
        float mag = magnitudes[i];
        uint8_t r = 128, g = 128, b = 128;  // Default gray
        
        if (mag < percentile_thresholds[0]) {
            // Below 5th percentile - very dark blue
            r = 0; g = 0; b = 100;
        } else if (mag >= percentile_thresholds.back()) {
            // Above 95th percentile - very dark red
            r = 100; g = 0; b = 0;
        } else {
            // Find which percentile range (5-10%, 10-15%, ..., 90-95%)
            bool found = false;
            for (size_t p = 0; p < percentile_thresholds.size() - 1; ++p) {
                if (mag >= percentile_thresholds[p] && mag < percentile_thresholds[p + 1]) {
                    // Map percentile index to color gradient (0.0 to 1.0)
                    double color_t = static_cast<double>(p) / (percentile_thresholds.size() - 1);
                    interpolate_color(color_t, r, g, b);
                    found = true;
                    break;
                }
            }
            // If not found in any range, use the last color (shouldn't happen, but safety check)
            if (!found) {
                interpolate_color(1.0, r, g, b);
            }
        }
        
        pcl_pt.r = r;
        pcl_pt.g = g;
        pcl_pt.b = b;
        colored_cloud->points.push_back(pcl_pt);
    }
    
    colored_cloud->width = colored_cloud->points.size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = false;
    
    std::cout << "Created colored point cloud with " << colored_cloud->points.size() << " points" << std::endl;

    // Visualize using PCL with white background
    std::cout << "Visualizing point cloud with percentile-based coloring..." << std::endl;
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("FPFH Percentile Colored Points"));
    viewer->setBackgroundColor(1.0, 1.0, 1.0);  // White background
    viewer->addPointCloud<pcl::PointXYZRGB>(colored_cloud, "colored_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "colored_cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    
    std::cout << "Press 'q' to quit the viewer" << std::endl;
    
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    return 0;
}

