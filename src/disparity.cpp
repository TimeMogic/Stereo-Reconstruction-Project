#include "disparity.h"
#include <opencv2/opencv.hpp>

cv::Mat computeDisparity(const cv::Mat& rectL, const cv::Mat& rectR) {
    // 1) Ensure input images are single-channel 8-bit (grayscale)
    cv::Mat L = rectL, R = rectR;
    if (L.channels() == 3) cv::cvtColor(L, L, cv::COLOR_BGR2GRAY);
    if (R.channels() == 3) cv::cvtColor(R, R, cv::COLOR_BGR2GRAY);

    if (L.type() != CV_8U) L.convertTo(L, CV_8U);
    if (R.type() != CV_8U) R.convertTo(R, CV_8U);

    // 2) SGBM key parameters
    int minDisp   = 0;
    int numDisp   = 240;   // Must be a multiple of 16; covers up to ~vmax=142
    int blockSize = 5;

    auto sgbm = cv::StereoSGBM::create(minDisp, numDisp, blockSize);

    int cn = 1; // Number of image channels (grayscale)
    sgbm->setP1(8  * cn * blockSize * blockSize);
    sgbm->setP2(32 * cn * blockSize * blockSize);

    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(2);
    sgbm->setDisp12MaxDiff(1);

    // Optional: limit pre-filtering to stabilize matching
    sgbm->setPreFilterCap(31);

    // 3) Compute disparity
    cv::Mat disp16;
    sgbm->compute(L, R, disp16);

    // StereoSGBM outputs CV_16S disparity scaled by 16
    cv::Mat disp32;
    disp16.convertTo(disp32, CV_32F, 1.0 / 16.0);

    return disp32;
}
