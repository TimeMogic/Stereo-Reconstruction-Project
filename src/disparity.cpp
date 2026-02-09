#include <limits>
#include <cmath>
#include <stdexcept>
#include "disparity.h"
#include <opencv2/opencv.hpp>

cv::Mat computeDisparityBM(const cv::Mat& rectL, const cv::Mat& rectR) {
    // Custom Block Matching (SAD + WTA + optional LR-consistency)
    cv::Mat L = rectL, R = rectR;
    if (L.empty() || R.empty()) {
        throw std::runtime_error("computeDisparityBM: empty input image(s)");
    }

    if (L.channels() == 3) cv::cvtColor(L, L, cv::COLOR_BGR2GRAY);
    if (R.channels() == 3) cv::cvtColor(R, R, cv::COLOR_BGR2GRAY);

    if (L.type() != CV_8U) L.convertTo(L, CV_8U);
    if (R.type() != CV_8U) R.convertTo(R, CV_8U);

    cv::equalizeHist(L, L);
    cv::equalizeHist(R, R);

    // For Middlebury Backpack, valid disparity range is roughly [vmin, vmax] = [32, 224].
    // Restricting the search range dramatically reduces false matches for simple BM.
    const int minDisp   = 32;
    const int maxDisp   = 224;
    const int numDisp   = (maxDisp - minDisp + 1);

    // Slightly larger window helps stability for SAD BM on real scenes.
    const int blockSize = 9;   // odd

    // Rejection/uniqueness heuristics (BM needs these)
    const int  maxCostPerPixel = 30;   // tune: 20~40
    const float uniqRatio      = 0.95f; // second_best must be >= best/uniqRatio

    if (blockSize <= 0 || (blockSize % 2) == 0) {
        throw std::runtime_error("computeDisparityBM: blockSize must be positive odd");
    }

    const int radius  = blockSize / 2;

    auto computeBM = [&](const cv::Mat& A, const cv::Mat& B) -> cv::Mat {
        cv::Mat disp(A.rows, A.cols, CV_32F,
                     cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

        for (int y = radius; y < A.rows - radius; ++y) {
            for (int x = radius; x < A.cols - radius; ++x) {
                int bestD          = minDisp;
                int bestCost       = std::numeric_limits<int>::max();
                int secondBestCost = std::numeric_limits<int>::max();

                for (int d = minDisp; d <= maxDisp; ++d) {
                    const int xb = x - d;
                    if (xb < radius || xb >= B.cols - radius) continue;

                    int cost = 0;
                    for (int wy = -radius; wy <= radius; ++wy) {
                        const uchar* aPtr = A.ptr<uchar>(y + wy);
                        const uchar* bPtr = B.ptr<uchar>(y + wy);
                        for (int wx = -radius; wx <= radius; ++wx) {
                            cost += std::abs(int(aPtr[x + wx]) - int(bPtr[xb + wx]));
                        }
                    }

                    if (cost < bestCost) {
                        secondBestCost = bestCost;
                        bestCost       = cost;
                        bestD          = d;
                    } else if (cost < secondBestCost) {
                        secondBestCost = cost;
                    }
                }

                // Reject ambiguous matches (uniqueness) and very high-cost matches (textureless/occluded)
                const int maxCost = maxCostPerPixel * blockSize * blockSize;
                const bool badCost = (bestCost > maxCost);

                // Uniqueness: if 2nd best is too close to best, likely ambiguous
                const bool ambiguous = (secondBestCost < std::numeric_limits<int>::max()) &&
                                       (float(secondBestCost) < float(bestCost) / uniqRatio);

                if (badCost || ambiguous) {
                    disp.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
                } else {
                    disp.at<float>(y, x) = static_cast<float>(bestD);
                }
            }
        }
        return disp;
    };

    // Left-to-right disparity
    cv::Mat dispL = computeBM(L, R);

    // Optional LR-consistency check
    const bool doLRCheck = false;
    const int  lrThresh  = 1; // pixels
    if (!doLRCheck) return dispL;

    // Right-to-left disparity
    cv::Mat dispR = computeBM(R, L);

    for (int y = 0; y < dispL.rows; ++y) {
        for (int x = 0; x < dispL.cols; ++x) {
            float dL = dispL.at<float>(y, x);
            if (!std::isfinite(dL)) continue;

            int xr = static_cast<int>(std::round(x - dL));
            if (xr < 0 || xr >= dispR.cols) {
                dispL.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            float dR = dispR.at<float>(y, xr);
            if (!std::isfinite(dR)) {
                dispL.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            float x_back = xr + dR;
            if (std::abs(x_back - x) > lrThresh) {
                dispL.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    return dispL;
}

cv::Mat computeDisparity(const cv::Mat& rectL, const cv::Mat& rectR) {
    // OpenCV SGBM baseline
    cv::Mat L = rectL, R = rectR;
    if (L.empty() || R.empty()) {
        throw std::runtime_error("computeDisparity: empty input image(s)");
    }

    if (L.channels() == 3) cv::cvtColor(L, L, cv::COLOR_BGR2GRAY);
    if (R.channels() == 3) cv::cvtColor(R, R, cv::COLOR_BGR2GRAY);

    if (L.type() != CV_8U) L.convertTo(L, CV_8U);
    if (R.type() != CV_8U) R.convertTo(R, CV_8U);

    const int minDisp   = 0;
    const int numDisp   = 240;
    const int blockSize = 5;

    auto sgbm = cv::StereoSGBM::create(minDisp, numDisp, blockSize);

    const int cn = 1;
    sgbm->setP1(8  * cn * blockSize * blockSize);
    sgbm->setP2(32 * cn * blockSize * blockSize);

    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(2);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setPreFilterCap(31);

    cv::Mat disp16;
    sgbm->compute(L, R, disp16);

    cv::Mat disp32;
    disp16.convertTo(disp32, CV_32F, 1.0 / 16.0);
    return disp32;
}
