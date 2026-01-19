#pragma once
#include <opencv2/opencv.hpp>

void estimatePose(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K,
    cv::Mat& R,
    cv::Mat& t
);
