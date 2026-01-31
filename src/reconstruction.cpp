#include "reconstruction.h"
#include <limits>

cv::Mat reconstructPointCloud(
    const cv::Mat& disparity,  //Cv_32F, d=xL-XR
    const cv::Mat& Q   //4x4, CV_64F (from stereoRectify)
) {
     CV_Assert(Q.rows == 4 && Q.cols == 4);
    CV_Assert(disparity.type() == CV_32F);

    cv::Mat points3D(disparity.size(), CV_32FC3);

    // 读出 Q 的元素（更快，也更清晰）
    const double q00 = Q.at<double>(0,0), q01 = Q.at<double>(0,1), q02 = Q.at<double>(0,2), q03 = Q.at<double>(0,3);
    const double q10 = Q.at<double>(1,0), q11 = Q.at<double>(1,1), q12 = Q.at<double>(1,2), q13 = Q.at<double>(1,3);
    const double q20 = Q.at<double>(2,0), q21 = Q.at<double>(2,1), q22 = Q.at<double>(2,2), q23 = Q.at<double>(2,3);
    const double q30 = Q.at<double>(3,0), q31 = Q.at<double>(3,1), q32 = Q.at<double>(3,2), q33 = Q.at<double>(3,3);

    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {

            float d = disparity.at<float>(y, x);

            // 无效视差（等价于 reprojectImageTo3D 的 handleMissingValues=true）
            if (!(d > 0.0f)) {
                points3D.at<cv::Vec3f>(y, x) = cv::Vec3f(
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN()
                );
                continue;
            }

            // Q * [x y d 1]^T
            const double Xh = q00 * x + q01 * y + q02 * d + q03;
            const double Yh = q10 * x + q11 * y + q12 * d + q13;
            const double Zh = q20 * x + q21 * y + q22 * d + q23;
            const double Wh = q30 * x + q31 * y + q32 * d + q33;

            if (Wh == 0.0) {
                points3D.at<cv::Vec3f>(y, x) = cv::Vec3f(
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN()
                );
                continue;
            }

            points3D.at<cv::Vec3f>(y, x) = cv::Vec3f(
                static_cast<float>(Xh / Wh),
                static_cast<float>(Yh / Wh),
                static_cast<float>(Zh / Wh)  //Xh/Wh为归一化过后的三维坐标X，Y，Z
            );
        }
    }

    return points3D;
}