#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cmath>
#include <stdexcept>

// project headers
#include "dataset.h"
#include "feature_matching.h"
#include "epipolar.h"
#include "rectification.h"
#include "disparity.h"
#include "evaluation.h"
#include "reconstruction.h"
#include "pfm_io.h"

// ---- PFM loader (Middlebury GT disparity: disp0.pfm / disp1.pfm) ----


int main() {
    try {
        // 1) Load dataset
        auto data = loadMiddlebury("./data/Backpack");
        std::cout << "Loaded images: "
                  << data.img_left.cols << " x "
                  << data.img_left.rows << std::endl;

        // 2) Feature matching
        std::vector<cv::Point2f> pts1, pts2;
        matchFeatures(data.img_left, data.img_right, pts1, pts2);
        std::cout << "Matched features: " << pts1.size() << std::endl;

        if (pts1.size() < 20) {
            throw std::runtime_error("Not enough matched features for pose estimation.");
        }

        // 3) Estimate relative pose (R, t) from essential matrix
        cv::Mat R, t;
        estimatePose(pts1, pts2, data.K, R, t);

        // 4) Rectify images and get Q
        cv::Mat rectL, rectR, Q;

        // ===== DISABLE RECTIFICATION (dataset already rectified) =====
        // stereoRectifyImages(
        //     data.img_left, data.img_right,
        //     data.K,
        //     R, t,
        //     rectL, rectR,
        //     Q
        // );

        // (optional) save rectified images for debugging
        std::filesystem::create_directories("output");
        // cv::imwrite("output/rectL.png", rectL);
        // cv::imwrite("output/rectR.png", rectR);

        // 5) Compute disparity on rectified images
        // cv::Mat disparity = computeDisparity(rectL, rectR);

        // ===== DISABLE RECTIFIED DISPARITY: compute on original images =====
        cv::Mat disparity = computeDisparity(data.img_left, data.img_right);

        double dmin_raw, dmax_raw;
        cv::minMaxLoc(disparity, &dmin_raw, &dmax_raw);
        std::cout << "Disparity raw range: " << dmin_raw << " ~ " << dmax_raw << std::endl;

        // 6) Convert disparity to float (robust)
        cv::Mat disparity_f;
        if (disparity.type() == CV_16S) {
            disparity.convertTo(disparity_f, CV_32F, 1.0 / 16.0);
        } else if (disparity.type() == CV_32F) {
            disparity_f = disparity.clone();
        } else {
            throw std::runtime_error("Unsupported disparity type");
        }

        double dmin, dmax;
        cv::minMaxLoc(disparity_f, &dmin, &dmax);
        std::cout << "Disparity float range: " << dmin << " ~ " << dmax << std::endl;

        // ===== Bad Pixel Ratio (Middlebury) =====
        // Middlebury GT disparity for left image is disp0.pfm
        cv::Mat disp_gt_f;
        try {
            const std::string gt_path1 = "./data/Backpack/disp0.pfm";
            const std::string gt_path2 = "../data/Backpack/disp0.pfm";
            const std::string gt_path3 = "/workspace/data/Backpack/disp0.pfm";

            if (std::filesystem::exists(gt_path1)) disp_gt_f = loadPFM(gt_path1);
            else if (std::filesystem::exists(gt_path2)) disp_gt_f = loadPFM(gt_path2);
            else if (std::filesystem::exists(gt_path3)) disp_gt_f = loadPFM(gt_path3);
            else {
                std::cout << "Bad Pixel Ratio: skipped (GT disp0.pfm not found)\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Bad Pixel Ratio: skipped (failed to load GT: " << e.what() << ")\n";
        }

        if (!disp_gt_f.empty()) {
            const float TAU = 4.0f;   // 4-pixel threshold (Middlebury standard)

            size_t bad = 0;
            size_t total = 0;

            // Use overlapping region in case sizes differ
            const int H = std::min(disparity_f.rows, disp_gt_f.rows);
            const int W = std::min(disparity_f.cols, disp_gt_f.cols);

            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float d_est = disparity_f.at<float>(y, x);
                    float d_gt  = disp_gt_f.at<float>(y, x);

                    // GT invalid pixels must be skipped (Middlebury convention)
                    if (!std::isfinite(d_gt) || d_gt <= 0) continue;
                    if (!std::isfinite(d_est) || d_est <= 0) continue;

                    total++;
                    if (std::abs(d_est - d_gt) > TAU) bad++;
                }
            }

            double bad_pixel_ratio = (total > 0) ? (double)bad / (double)total : 0.0;

            std::cout << "Bad Pixel Ratio (tau = " << TAU << " px): "
                      << bad_pixel_ratio * 100.0 << " % ("
                      << bad << " / " << total << ")"
                      << std::endl;
        }

        // 7) Save disparity visualization
        if (dmax > 0) {
            cv::Mat disp_vis;
            disparity_f.convertTo(disp_vis, CV_8U, 255.0 / dmax);
            cv::imwrite("output/disparity.png", disp_vis);
        }

        // 8) Stereo params (from Middlebury calib)
        const double fx = data.K.at<double>(0,0);
        const double fy = data.K.at<double>(1,1);
        const double cx = data.K.at<double>(0,2);
        const double cy = data.K.at<double>(1,2);
        const double B_mm = data.baseline; // baseline in mm

        std::cout << "fx=" << fx << " fy=" << fy
                  << " cx=" << cx << " cy=" << cy
                  << " baseline(mm)=" << B_mm
                  << " doffs=" << data.doffs
                  << " vmin=" << data.vmin
                  << " vmax=" << data.vmax
                  << std::endl;

        // 9) Reconstruct colored point cloud and save to PLY
        ReconstructionParams rp;
        rp.vmin = (float)data.vmin;
        rp.vmax = (float)data.vmax;
        rp.doffs = (float)data.doffs;
        rp.z_min_m = 0.2f;
        rp.z_max_m = 50.0f;
        rp.center = true;

        auto rr = saveColoredPointCloudPLY(
            "output/points.ply",
            disparity_f,
            data.img_left_color,
            data.K,
            data.baseline,
            rp
        );

        std::cout << "Centroid (m): mx=" << rr.centroid_m[0]
          << " my=" << rr.centroid_m[1]
          << " mz=" << rr.centroid_m[2] << "\n";

        std::cout << "Saved colored point cloud with "
          << rr.valid_points
          << " points to output/points.ply\n";

        std::cout << "Done.\n";
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error:\n" << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "STD error:\n" << e.what() << std::endl;
        return -1;
    }

    return 0;
}
