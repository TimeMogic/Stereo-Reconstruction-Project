#pragma once
#include <opencv2/opencv.hpp>

cv::Mat computeDisparity(
    const cv::Mat& rectL,
    const cv::Mat& rectR
);

// Our custom Block Matching disparity (SAD + WTA + optional LR-check)
cv::Mat computeDisparityBM(
    const cv::Mat& rectL,
    const cv::Mat& rectR
);
