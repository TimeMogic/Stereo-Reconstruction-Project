#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

void matchFeatures(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2
);
