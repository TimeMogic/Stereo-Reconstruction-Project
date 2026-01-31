#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// Load PFM file (Pf / PF) into CV_32FC1 or CV_32FC3
// - Supports Middlebury disp0.pfm / disp1.pfm
// - Throws std::runtime_error on failure
cv::Mat loadPFM(const std::string& path);
