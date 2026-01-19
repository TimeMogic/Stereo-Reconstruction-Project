#include "feature_matching.h"
#include <opencv2/features2d.hpp>

void matchFeatures(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2
) {
    // SIFT works best on grayscale
    cv::Mat g1, g2;
    if (img1.channels() == 3) cv::cvtColor(img1, g1, cv::COLOR_BGR2GRAY);
    else g1 = img1;
    if (img2.channels() == 3) cv::cvtColor(img2, g2, cv::COLOR_BGR2GRAY);
    else g2 = img2;

    // 1) Create SIFT
    auto sift = cv::SIFT::create(2000);

    // 2) Detect + compute
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    sift->detectAndCompute(g1, cv::noArray(), kp1, desc1);
    sift->detectAndCompute(g2, cv::noArray(), kp2, desc2);

    pts1.clear();
    pts2.clear();
    if (desc1.empty() || desc2.empty()) return;

    // 3) Match: SIFT descriptors are float => L2
    cv::BFMatcher matcher(cv::NORM_L2);

    // KNN + Lowe ratio test (more robust)
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc1, desc2, knn, 2);

    const float ratio = 0.75f;
    for (const auto& m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance < ratio * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
}
