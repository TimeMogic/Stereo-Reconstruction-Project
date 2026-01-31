#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <cstddef>

struct ReconstructionParams {
    float vmin = 0.f;      // disparity range filter (estimated disparity)
    float vmax = 0.f;
    float doffs = 0.f;     // disparity offset used for geometry: d_geom = d_est + doffs
    float z_min_m = 0.2f;  // depth range filter in meters
    float z_max_m = 50.0f;
    bool center = true;    // subtract centroid before writing
};

struct ReconstructionResult {
    std::size_t valid_points = 0;
    cv::Vec3d centroid_m = {0.0, 0.0, 0.0};
};

// Save colored point cloud to ASCII PLY.
// Inputs:
// - disparity_f: CV_32F disparity in pixels (already scaled properly, e.g. SGBM /16)
// - img_left_color: CV_8UC3 BGR image (same size as disparity)
// - K: 3x3 CV_64F intrinsics (fx, fy, cx, cy)
// - baseline_mm: baseline in millimeters (Middlebury calib)
// - params: filters + doffs + centering
//
// Returns:
// - number of valid points written and centroid (in meters)
//
// Throws std::runtime_error on errors (e.g. file open, invalid inputs)
ReconstructionResult saveColoredPointCloudPLY(
    const std::string& ply_path,
    const cv::Mat& disparity_f,
    const cv::Mat& img_left_color,
    const cv::Mat& K,
    double baseline_mm,
    const ReconstructionParams& params
);