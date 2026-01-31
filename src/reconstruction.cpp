#include "reconstruction.h"

#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

static inline bool finite_pos(float v) {
    return std::isfinite(v) && (v > 0.0f);
}

ReconstructionResult saveColoredPointCloudPLY(
    const std::string& ply_path,
    const cv::Mat& disparity_f,
    const cv::Mat& img_left_color,
    const cv::Mat& K,
    double baseline_mm,
    const ReconstructionParams& params
) {
    // ---- Input checks ----
    if (disparity_f.empty()) {
        throw std::runtime_error("saveColoredPointCloudPLY: disparity_f is empty");
    }
    if (img_left_color.empty()) {
        throw std::runtime_error("saveColoredPointCloudPLY: img_left_color is empty");
    }
    if (disparity_f.type() != CV_32F) {
        throw std::runtime_error("saveColoredPointCloudPLY: disparity_f must be CV_32F");
    }
    if (img_left_color.type() != CV_8UC3) {
        throw std::runtime_error("saveColoredPointCloudPLY: img_left_color must be CV_8UC3 (BGR)");
    }
    if (K.empty() || K.rows != 3 || K.cols != 3 || K.type() != CV_64F) {
        throw std::runtime_error("saveColoredPointCloudPLY: K must be 3x3 CV_64F");
    }

    // Use overlapping region if sizes differ (robust)
    const int H = std::min(disparity_f.rows, img_left_color.rows);
    const int W = std::min(disparity_f.cols, img_left_color.cols);
    if (H <= 0 || W <= 0) {
        throw std::runtime_error("saveColoredPointCloudPLY: invalid overlap size");
    }

    const double fx = K.at<double>(0,0);
    const double fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2);
    const double cy = K.at<double>(1,2);

    if (fx <= 0.0 || fy <= 0.0 || baseline_mm <= 0.0) {
        throw std::runtime_error("saveColoredPointCloudPLY: invalid intrinsics or baseline");
    }

    const float VMIN  = params.vmin;
    const float VMAX  = params.vmax;
    const float DOFFS = params.doffs;
    const float Z_MIN_M = params.z_min_m;
    const float Z_MAX_M = params.z_max_m;

    const bool use_disp_range = (std::isfinite(VMIN) && std::isfinite(VMAX) && (VMAX > VMIN));

    // ---- Pass 1: count valid points and compute centroid ----
    std::size_t valid_count = 0;
    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;

    for (int y = 0; y < H; ++y) {
        const float* drow = disparity_f.ptr<float>(y);

        for (int x = 0; x < W; ++x) {
            const float d_est = drow[x];
            if (!finite_pos(d_est)) continue;

            if (use_disp_range) {
                if (d_est < VMIN || d_est > VMAX) continue;
            }

            const float d_geom = d_est + DOFFS;
            if (!(d_geom > 0.0f) || !std::isfinite(d_geom)) continue;

            const float Z_mm = (float)(fx * baseline_mm / (double)d_geom);
            const float Z_m  = Z_mm * 0.001f;
            if (!std::isfinite(Z_m)) continue;
            if (Z_m < Z_MIN_M || Z_m > Z_MAX_M) continue;

            const float X_m = (float)(((x - cx) * (double)Z_mm / fx) * 0.001);
            const float Y_m = (float)(((y - cy) * (double)Z_mm / fy) * 0.001);

            if (!std::isfinite(X_m) || !std::isfinite(Y_m)) continue;

            sumX += (double)X_m;
            sumY += (double)Y_m;
            sumZ += (double)Z_m;
            valid_count++;
        }
    }

    ReconstructionResult res;
    res.valid_points = valid_count;

    double mx = 0.0, my = 0.0, mz = 0.0;
    if (valid_count > 0) {
        mx = sumX / (double)valid_count;
        my = sumY / (double)valid_count;
        mz = sumZ / (double)valid_count;
    }
    res.centroid_m = cv::Vec3d(mx, my, mz);

    // ---- Open PLY file ----
    std::ofstream ofs(ply_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("saveColoredPointCloudPLY: failed to open " + ply_path);
    }

    // ---- Write header ----
    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << valid_count << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";

    // ---- Pass 2: write points ----
    for (int y = 0; y < H; ++y) {
        const float* drow = disparity_f.ptr<float>(y);

        for (int x = 0; x < W; ++x) {
            const float d_est = drow[x];
            if (!finite_pos(d_est)) continue;

            if (use_disp_range) {
                if (d_est < VMIN || d_est > VMAX) continue;
            }

            const float d_geom = d_est + DOFFS;
            if (!(d_geom > 0.0f) || !std::isfinite(d_geom)) continue;

            const float Z_mm = (float)(fx * baseline_mm / (double)d_geom);
            const float Z_m  = Z_mm * 0.001f;
            if (!std::isfinite(Z_m)) continue;
            if (Z_m < Z_MIN_M || Z_m > Z_MAX_M) continue;

            const float X_m = (float)(((x - cx) * (double)Z_mm / fx) * 0.001);
            const float Y_m = (float)(((y - cy) * (double)Z_mm / fy) * 0.001);

            if (!std::isfinite(X_m) || !std::isfinite(Y_m)) continue;

            // color from left color image (BGR)
            const cv::Vec3b c = img_left_color.at<cv::Vec3b>(y, x);

            float Xc = X_m;
            float Yc = Y_m;
            float Zc = Z_m;
            if (params.center) {
                Xc -= (float)mx;
                Yc -= (float)my;
                Zc -= (float)mz;
            }

            ofs << Xc << " " << Yc << " " << Zc << " "
                << (int)c[2] << " "  // R
                << (int)c[1] << " "  // G
                << (int)c[0] << "\n"; // B
        }
    }

    ofs.close();
    return res;
}
