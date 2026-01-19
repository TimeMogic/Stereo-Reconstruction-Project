#include "rectification.h"

void stereoRectifyImages(
    const cv::Mat& imgL,
    const cv::Mat& imgR,
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t,
    cv::Mat& rectL,
    cv::Mat& rectR,
    cv::Mat& Q
) {
    cv::Mat R1, R2, P1, P2;
    cv::stereoRectify(
        K, cv::Mat(), K, cv::Mat(),
        imgL.size(), R, t,
        R1, R2, P1, P2, Q
    );

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(
        K, cv::Mat(), R1, P1, imgL.size(), CV_32FC1, map1x, map1y
    );
    cv::initUndistortRectifyMap(
        K, cv::Mat(), R2, P2, imgR.size(), CV_32FC1, map2x, map2y
    );

    cv::remap(imgL, rectL, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(imgR, rectR, map2x, map2y, cv::INTER_LINEAR);
}
