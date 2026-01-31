Stereo Reconstruction (Docker)

本项目是一个 基于双目视觉的 3D 重建（Stereo Reconstruction） 示例，
使用 C++ + OpenCV 实现，并通过 Docker 封装，保证在 Windows / macOS / Linux 上可复现运行。

一、你能得到什么

运行完成后，会在 output/ 目录下生成：

disparity.png：双目视差图

points.ply：彩色 3D 点云（可用 MeshLab  打开）

二、运行环境（只需要这个）

Docker Desktop

Windows / macOS / Linux 均可

下载地址：https://www.docker.com/products/docker-desktop/

⚠️ 不需要安装 OpenCV / CMake / 编译器

三、项目结构

```
stereo_reconstruction/
├── Dockerfile
├── CMakeLists.txt
├── include/
├── src/
├── data/ #需要从链接中下载到文件夹下
│   └── （比如）artroom1/
│       ├── im0.png
│       ├── im1.png
│       └── calib.txt
├── output/        # 运行后自动生成
└── README.md
```

四、数据集下载链接

[Middlebury Stereo Dataset - Backpack-imperfect](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Backpack-imperfect/)

下载完数据集解压之后，请放在 `data` 文件夹中。

五、如何运行（3 步）

1️⃣ 进入项目目录
```bash
cd stereo_reconstruction
```

2️⃣ 构建 Docker 镜像（只需一次）
```bash
docker build -t stereo_recon .
```

3️⃣ 运行程序（推荐命令）
```bash
docker run --rm \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/output:/workspace/output" \
  stereo_recon
```

Windows PowerShell 用户请使用：
```powershell
docker run --rm \`
  -v ${PWD}/data:/workspace/data \`
  -v ${PWD}/output:/workspace/output\ `
  stereo_recon
```

六、查看结果
1️⃣ 查看视差图
```plaintext
output/disparity.png
```

2️⃣ 查看点云

打开以下文件：
```plaintext
output/points.ply
```

推荐软件：

MeshLab


如果打开后什么也看不到，请点击：

MeshLab：View → Reset Trackball 或 Fit to Screen
