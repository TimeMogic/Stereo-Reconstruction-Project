#include "dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

StereoDataset loadMiddlebury(const std::string& scene_dir) {
    StereoDataset data;

    const std::string left_path  = scene_dir + "/im0.png";
    const std::string right_path = scene_dir + "/im1.png";
    const std::string calib_path = scene_dir + "/calib.txt";

    // Load color images (BGR)
    cv::Mat left_color  = cv::imread(left_path,  cv::IMREAD_COLOR);
    cv::Mat right_color = cv::imread(right_path, cv::IMREAD_COLOR);

    if (left_color.empty() || right_color.empty()) {
        throw std::runtime_error("Failed to load stereo images from " + scene_dir);
    }

    // Keep left color image for point cloud coloring
    data.img_left_color = left_color;

    // Convert to grayscale for matching / disparity computation
    cv::cvtColor(left_color,  data.img_left,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_color, data.img_right, cv::COLOR_BGR2GRAY);

    if (data.img_left.empty() || data.img_right.empty()) {
        throw std::runtime_error("Failed to load stereo images from " + scene_dir);
    }

    std::ifstream fin(calib_path);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open calib.txt in " + scene_dir);
    }

    data.K = cv::Mat::eye(3, 3, CV_64F);
    data.baseline = -1.0;
    data.doffs = 0.0;
    data.vmin = 0.0;
    data.vmax = 0.0;

    std::string line;
    while (std::getline(fin, line)) {

        // cam0=[a b c; d e f; g h i]
        if (starts_with(line, "cam0=")) {
            auto l = line.find('[');
            auto r = line.find(']');
            if (l == std::string::npos || r == std::string::npos) {
                throw std::runtime_error("Invalid cam0 format in calib.txt");
            }

            std::string content = line.substr(l + 1, r - l - 1);
            std::replace(content.begin(), content.end(), ';', ' ');

            std::stringstream ss(content);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (!(ss >> data.K.at<double>(i, j))) {
                        throw std::runtime_error("Failed to parse cam0 matrix");
                    }
                }
            }
        }

        if (starts_with(line, "baseline=")) {
            data.baseline = std::stod(line.substr(std::string("baseline=").size()));
        }

        if (starts_with(line, "doffs=")) {
            data.doffs = std::stod(line.substr(std::string("doffs=").size()));
        }

        if (starts_with(line, "vmin=")) {
            data.vmin = std::stod(line.substr(std::string("vmin=").size()));
        }

        if (starts_with(line, "vmax=")) {
            data.vmax = std::stod(line.substr(std::string("vmax=").size()));
        }
    }

    fin.close();

    // Sanity check
    const double fx = data.K.at<double>(0,0);
    const double fy = data.K.at<double>(1,1);

    if (fx <= 0 || fy <= 0 || data.baseline <= 0) {
        std::ostringstream oss;
        oss << "Invalid calibration parsed.\n"
            << "fx=" << fx << " fy=" << fy
            << " baseline=" << data.baseline
            << " doffs=" << data.doffs
            << " vmin=" << data.vmin
            << " vmax=" << data.vmax;
        throw std::runtime_error(oss.str());
    }

    return data;
}
