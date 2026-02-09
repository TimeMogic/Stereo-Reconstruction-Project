#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <cstddef>

struct MiddleburyEvalResult {
    bool has_gt = false;

    // 统计像素数量
    std::size_t total = 0;      // valid pixels compared
    std::size_t bad_raw = 0;    // |d_est - d_gt| > tau

    // 结果指标（老师需要的）
    double bpr_raw_percent = 0.0; // Bad Pixel Ratio (%)
    double rmse_raw = 0.0;        // RMSE (px)
};

// 读 Middlebury 的 PFM（disp0.pfm/disp1.pfm），返回 CV_32FC1 或 CV_32FC3
cv::Mat loadPFM(const std::string& path);

// 尝试在多个候选路径里找到 disp0.pfm，找到了就读；都没有则返回空 Mat
cv::Mat tryLoadDisp0PFM(const std::string& scene_dir);

// Middlebury evaluation (简化版)：只计算 BPR raw 和 RMSE raw
// disparity_f: CV_32F disparity (pixel units)
// disp_gt_f:  CV_32F GT disparity (pixel units)
// vmin/vmax:  recommended disparity range; 传 NaN 或 vmax<=vmin 可禁用过滤
// tau:        bad pixel threshold (in pixels)
MiddleburyEvalResult evaluateMiddleburyDisparity(
    const cv::Mat& disparity_f,
    const cv::Mat& disp_gt_f,
    float vmin,
    float vmax,
    float tau = 4.0f,
    bool use_vmin_vmax_filter = false
);
