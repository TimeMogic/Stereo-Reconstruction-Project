FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1) Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) Copy source
WORKDIR /workspace
COPY . .

# 3) Build (auto-detect OpenCV_DIR for different CPU arch paths)
RUN OpenCV_DIR="$(dirname "$(dpkg -L libopencv-dev | grep -m1 OpenCVConfig.cmake)")" && \
    echo "Detected OpenCV_DIR=${OpenCV_DIR}" && \
    mkdir -p build && cd build && \
    cmake .. -DOpenCV_DIR="${OpenCV_DIR}" && \
    make -j"$(nproc)"

# 4) Run
CMD ["./build/stereo_reconstruction"]
