#pragma once
#include <opencv2/opencv.hpp>

cv::Mat computeDisparity(
    const cv::Mat& rectL,
    const cv::Mat& rectR
);
