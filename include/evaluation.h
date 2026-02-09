#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <cstddef>

struct MiddleburyEvalResult {
    bool has_gt = false;

    // Pixel statistics
    std::size_t total = 0;      // valid pixels compared
    std::size_t bad_raw = 0;    // |d_est - d_gt| > tau

    // Evaluation metrics (required)
    double bpr_raw_percent = 0.0; // Bad Pixel Ratio (%)
    double rmse_raw = 0.0;        // RMSE (px)
};

// Load Middlebury PFM disparity file (disp0.pfm / disp1.pfm)
// Returns CV_32FC1 or CV_32FC3
cv::Mat loadPFM(const std::string& path);

// Try to locate and load disp0.pfm from multiple candidate paths
// Returns empty Mat if not found
cv::Mat tryLoadDisp0PFM(const std::string& scene_dir);

// Middlebury evaluation (simplified): compute only BPR raw and RMSE raw
// disparity_f: CV_32F estimated disparity (pixel units)
// disp_gt_f:   CV_32F ground-truth disparity (pixel units)
// vmin/vmax:   recommended disparity range; pass NaN or vmax <= vmin to disable filtering
// tau:         bad pixel threshold (in pixels)
MiddleburyEvalResult evaluateMiddleburyDisparity(
    const cv::Mat& disparity_f,
    const cv::Mat& disp_gt_f,
    float vmin,
    float vmax,
    float tau = 4.0f,
    bool use_vmin_vmax_filter = false
);
