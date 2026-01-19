#include "reconstruction.h"

cv::Mat reconstructPointCloud(
    const cv::Mat& disparity,
    const cv::Mat& Q
) {
    cv::Mat points3D;
    cv::reprojectImageTo3D(
        disparity,
        points3D,
        Q,
        true   // ⭐handleMissingValues = true
    );
    return points3D;
}
