#pragma once
#include <opencv2/opencv.hpp>

void stereoRectifyImages(
    const cv::Mat& imgL,
    const cv::Mat& imgR,
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t,
    cv::Mat& rectL,
    cv::Mat& rectR,
    cv::Mat& Q
);
