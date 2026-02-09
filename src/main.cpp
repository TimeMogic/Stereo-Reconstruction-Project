#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cmath>
#include <stdexcept>
#include <chrono>
using Clock = std::chrono::steady_clock;
using ms = std::chrono::duration<double, std::milli>;

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
        auto t_disp_cv_start = Clock::now();
        cv::Mat disparity = computeDisparity(data.img_left, data.img_right);
        auto t_disp_cv_end = Clock::now();

        auto t_disp_ours_start = Clock::now();
        cv::Mat disparity_ours = computeDisparityBM(data.img_left, data.img_right);
        auto t_disp_ours_end = Clock::now();

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

        cv::Mat disparity_ours_f;
        if (disparity_ours.type() == CV_16S) {
            disparity_ours.convertTo(disparity_ours_f, CV_32F, 1.0 / 16.0);
        } else if (disparity_ours.type() == CV_32F) {
            disparity_ours_f = disparity_ours.clone();
        }

        double dmin, dmax;
        cv::minMaxLoc(disparity_f, &dmin, &dmax);
        std::cout << "Disparity float range: " << dmin << " ~ " << dmax << std::endl;

        // ===== Bad Pixel Ratio (Middlebury) =====
        
        // ===== Middlebury Evaluation (module) =====
    {
        const std::string scene_dir = "./data/Backpack";

        cv::Mat disp_gt_f = tryLoadDisp0PFM(scene_dir);
        if (disp_gt_f.empty()) {
            std::cout << "Evaluation: skipped (GT disp0.pfm not found or failed to load)\n";
        } else {
            const float TAU = 4.0f;

            auto eval = evaluateMiddleburyDisparity(
                disparity_f,
                disp_gt_f,
                (float)data.vmin,
                (float)data.vmax,
                TAU,
                /*use_vmin_vmax_filter=*/false
            );

            std::cout << "Eval (tau=" << TAU << "px)\n";
            std::cout << "  Valid pixels: " << eval.total << "\n";
            std::cout << "  BPR raw   : " << eval.bpr_raw_percent
                      << " % (" << eval.bad_raw << " / " << eval.total << ")\n";
            std::cout << "  RMSE raw  : " << eval.rmse_raw << " px\n";

            auto eval_ours = evaluateMiddleburyDisparity(
                disparity_ours_f,
                disp_gt_f,
                (float)data.vmin,
                (float)data.vmax,
                TAU,
                /*use_vmin_vmax_filter=*/false
            );

            std::cout << "Eval OURS (BM) (tau=" << TAU << "px)\n";
            std::cout << "  Valid pixels: " << eval_ours.total << "\n";
            std::cout << "  BPR raw   : " << eval_ours.bpr_raw_percent
                      << " % (" << eval_ours.bad_raw << " / " << eval_ours.total << ")\n";
            std::cout << "  RMSE raw  : " << eval_ours.rmse_raw << " px\n";
        }
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

        auto t_rec_cv_start = Clock::now();
        auto rr = saveColoredPointCloudPLY(
            "output/points.ply",
            disparity_f,
            data.img_left_color,
            data.K,
            data.baseline,
            rp
        );
        auto t_rec_cv_end = Clock::now();

        auto t_rec_ours_start = Clock::now();
        auto rr_ours = saveColoredPointCloudPLY_OURS(
            "output/points_ours.ply",
            disparity_ours_f,
            data.img_left_color,
            data.K,
            data.baseline, // replace with your custom baseline if needed
            rp
        );
        auto t_rec_ours_end = Clock::now();

        std::cout << "Centroid (m): mx=" << rr.centroid_m[0]
          << " my=" << rr.centroid_m[1]
          << " mz=" << rr.centroid_m[2] << "\n";

        std::cout << "Saved colored point cloud with "
          << rr.valid_points
          << " points to output/points.ply\n";

        std::cout << "\n===== Runtime =====\n";
        std::cout << "[OpenCV] Disparity: "
                  << ms(t_disp_cv_end - t_disp_cv_start).count() << " ms\n";
        std::cout << "[OpenCV] Reconstruction: "
                  << ms(t_rec_cv_end - t_rec_cv_start).count() << " ms\n";

        std::cout << "[OURS] Disparity: "
                  << ms(t_disp_ours_end - t_disp_ours_start).count() << " ms\n";
        std::cout << "[OURS] Reconstruction: "
                  << ms(t_rec_ours_end - t_rec_ours_start).count() << " ms\n";

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
