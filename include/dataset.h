#pragma once
#include <string>
#include <opencv2/opencv.hpp>

struct StereoDataset {
    cv::Mat img_left;        // grayscale for matching
    cv::Mat img_right;       // grayscale for matching

    cv::Mat img_left_color;  // color for point cloud coloring (BGR)

    // Intrinsics (cam0)
    cv::Mat K;               // 3x3 CV_64F

    // Middlebury calib params
    double baseline = -1.0;  // usually in mm
    double doffs    = 0.0;   // disparity offset
    double vmin     = 0.0;   // valid disparity min
    double vmax     = 0.0;   // valid disparity max
};

StereoDataset loadMiddlebury(const std::string& scene_dir);
