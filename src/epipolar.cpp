#include "epipolar.h"

void estimatePose(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K,
    cv::Mat& R,
    cv::Mat& t
) {
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K);
    cv::recoverPose(E, pts1, pts2, K, R, t);
}

