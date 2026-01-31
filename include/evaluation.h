#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <cstddef>

struct MiddleburyEvalResult {
    bool has_gt = false;

    // 统计像素数量
    std::size_t total = 0;
    std::size_t bad_raw = 0;   // |d_est - d_gt| > tau
    std::size_t bad_geom = 0;  // |(d_est + doffs) - d_gt| > tau

    // 误差统计（可选）
    double mae_raw = 0.0;
    double rmse_raw = 0.0;
    double mae_geom = 0.0;
    double rmse_geom = 0.0;

    // 最终比例（%）
    double bpr_raw_percent = 0.0;
    double bpr_geom_percent = 0.0;
};

// 读 Middlebury 的 PFM（disp0.pfm/disp1.pfm），返回 CV_32FC1 或 CV_32FC3
cv::Mat loadPFM(const std::string& path);

// 尝试在多个候选路径里找到 disp0.pfm，找到了就读；都没有则返回空 Mat
cv::Mat tryLoadDisp0PFM(const std::string& scene_dir);

// Middlebury evaluation：Bad Pixel Ratio + (可选) MAE/RMSE
// disparity_f: CV_32F disparity (pixel units)
// disp_gt_f:  CV_32F GT disparity (pixel units)
// doffs:      disparity offset in pixels
// vmin/vmax:  recommended disparity range; 传 NaN 或 vmax<=vmin 可禁用过滤
MiddleburyEvalResult evaluateMiddleburyDisparity(
    const cv::Mat& disparity_f,
    const cv::Mat& disp_gt_f,
    float doffs,
    float vmin,
    float vmax,
    float tau = 4.0f,
    bool use_vmin_vmax_filter = false,
    bool compute_mae_rmse = true
);
