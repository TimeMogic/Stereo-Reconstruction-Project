#include "pfm_io.h"

#include <fstream>
#include <stdexcept>

// ---- PFM loader (Middlebury GT disparity: disp0.pfm / disp1.pfm) ----
cv::Mat loadPFM(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open PFM: " + path);

    std::string type;
    ifs >> type; // "Pf" or "PF"
    int w = 0, h = 0;
    ifs >> w >> h;

    float scale = 0.f;
    ifs >> scale;
    ifs.get(); // consume single whitespace/newline after header

    const bool color = (type == "PF");
    if (type != "Pf" && type != "PF") {
        throw std::runtime_error("Invalid PFM header (expect Pf/PF): " + path);
    }

    const int channels = color ? 3 : 1;
    cv::Mat img(h, w, (channels == 1) ? CV_32FC1 : CV_32FC3);

    // PFM stores rows bottom-to-top
    for (int y = h - 1; y >= 0; --y) {
        ifs.read(reinterpret_cast<char*>(img.ptr<float>(y)),
                 sizeof(float) * w * channels);
        if (!ifs) throw std::runtime_error("PFM read failed: " + path);
    }

    // scale sign indicates endianness (negative = little endian).
    // Most Middlebury PFM are little-endian; on x86 little-endian machines,
    // we typically don't need to byte-swap.
    return img;
}
