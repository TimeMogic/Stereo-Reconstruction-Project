#include "evaluation.h"
#include "pfm_io.h"

#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <cmath>
#include <algorithm>



cv::Mat tryLoadDisp0PFM(const std::string& scene_dir) {
    const std::string p1 = scene_dir + "/disp0.pfm";
    const std::string p2 = scene_dir + "/../" + std::filesystem::path(scene_dir).filename().string() + "/disp0.pfm";
    const std::string p3 = "/workspace" + scene_dir + "/disp0.pfm"; // optional Docker layout

    try {
        if (std::filesystem::exists(p1)) return loadPFM(p1);
        if (std::filesystem::exists(p2)) return loadPFM(p2);
        if (std::filesystem::exists(p3)) return loadPFM(p3);
    } catch (...) {
        // swallow; caller will handle empty or throw if needed
    }
    return cv::Mat();
}

static inline bool finite_pos(float v) {
    return std::isfinite(v) && (v > 0.0f);
}

MiddleburyEvalResult evaluateMiddleburyDisparity(
    const cv::Mat& disparity_f,
    const cv::Mat& disp_gt_f,
    float doffs,
    float vmin,
    float vmax,
    float tau,
    bool use_vmin_vmax_filter,
    bool compute_mae_rmse
) {
    MiddleburyEvalResult res;

    if (disparity_f.empty() || disp_gt_f.empty()) {
        return res; // has_gt = false
    }
    if (disparity_f.type() != CV_32F) {
        throw std::runtime_error("evaluateMiddleburyDisparity: disparity_f must be CV_32F");
    }
    if (disp_gt_f.type() != CV_32F) {
        throw std::runtime_error("evaluateMiddleburyDisparity: disp_gt_f must be CV_32F (Pf)");
    }

    res.has_gt = true;

    const int H = std::min(disparity_f.rows, disp_gt_f.rows);
    const int W = std::min(disparity_f.cols, disp_gt_f.cols);

    double sum_abs_raw = 0.0, sum_sq_raw = 0.0;
    double sum_abs_geom = 0.0, sum_sq_geom = 0.0;

    const bool enable_range_filter = use_vmin_vmax_filter && (vmax > vmin) && std::isfinite(vmin) && std::isfinite(vmax);

    for (int y = 0; y < H; ++y) {
        const float* est_row = disparity_f.ptr<float>(y);
        const float* gt_row  = disp_gt_f.ptr<float>(y);

        for (int x = 0; x < W; ++x) {
            const float d_est = est_row[x];
            const float d_gt  = gt_row[x];

            // GT invalid pixels must be skipped (Middlebury convention)
            if (!finite_pos(d_gt)) continue;
            if (!finite_pos(d_est)) continue;

            if (enable_range_filter) {
                if (d_est < vmin || d_est > vmax) continue;
            }

            res.total++;

            const float err_raw  = std::abs(d_est - d_gt);
            const float d_geom   = d_est + doffs;
            const float err_geom = std::abs(d_geom - d_gt);

            if (err_raw > tau)  res.bad_raw++;
            if (err_geom > tau) res.bad_geom++;

            if (compute_mae_rmse) {
                sum_abs_raw  += err_raw;
                sum_sq_raw   += (double)err_raw * (double)err_raw;
                sum_abs_geom += err_geom;
                sum_sq_geom  += (double)err_geom * (double)err_geom;
            }
        }
    }

    if (res.total > 0) {
        res.bpr_raw_percent  = 100.0 * (double)res.bad_raw  / (double)res.total;
        res.bpr_geom_percent = 100.0 * (double)res.bad_geom / (double)res.total;

        if (compute_mae_rmse) {
            res.mae_raw  = sum_abs_raw  / (double)res.total;
            res.rmse_raw = std::sqrt(sum_sq_raw / (double)res.total);

            res.mae_geom  = sum_abs_geom / (double)res.total;
            res.rmse_geom = std::sqrt(sum_sq_geom / (double)res.total);
        }
    }

    return res;
}
