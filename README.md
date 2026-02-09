# Stereo Reconstruction (Docker)

This project is a **Stereo Vision–based 3D Reconstruction** demo implemented in **C++ with OpenCV** and fully packaged with **Docker**. The Docker setup ensures the project can be built and run **reproducibly** on **Windows, macOS, and Linux**, without requiring any local dependency installation.

**Link**: https://github.com/TimeMogic/Stereo-Reconstruction-Project



---

## 1. What You Will Get

After a successful run, the following outputs will be generated in the `output/` directory:

* **`disparity.png`** – the computed stereo disparity map
* **`points.ply`** – a colored 3D point cloud (can be opened with MeshLab)

---

## 2. Runtime Requirements (Only This Is Needed)

* **Docker Desktop**

  * Supported on Windows / macOS / Linux
  * Download: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

⚠️ You **do not** need to install OpenCV, CMake, or any compiler locally.

---

## 3. Project Structure

```text
stereo_reconstruction/
├── Dockerfile
├── CMakeLists.txt
├── include/
├── src/
├── data/            # Download the dataset and place it here
│   └── (e.g.) artroom1/
│       ├── im0.png
│       ├── im1.png
│       └── calib.txt
├── output/          # Automatically generated after running
└── README.md
```

---

## 4. Dataset Download

This project uses the **Middlebury Stereo Dataset**.

**Recommended dataset:**

* Backpack (imperfect)
* Link: [https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Backpack-imperfect/](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Backpack-imperfect/)

After downloading and extracting the dataset, place it under the `data/` directory, for example:

```text
data/Backpack-imperfect/
```

---

## 5. How to Run (3 Steps)

### Step 1: Enter the Project Directory

```bash
cd stereo_reconstruction
```

### Step 2: Build the Docker Image (Only Once)

```bash
docker build -t stereo_recon .
```

### Step 3: Run the Program (Recommended)

```bash
docker run --rm \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/output:/workspace/output" \
  stereo_recon
```

#### Windows (PowerShell)

```powershell
docker run --rm `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/output:/workspace/output `
  stereo_recon
```

---

## 6. Viewing the Results

### 1️⃣ Disparity Map

Open the following file:

```text
output/disparity.png
```

### 2️⃣ 3D Point Cloud

Open the following file:

```text
output/points.ply
```

**Recommended viewer:**

* **MeshLab**

If nothing is visible after opening the point cloud, try:

* `View → Reset Trackball`
* or `View → Fit to Screen`

---

## Notes

* This project is intended as a **clean, minimal stereo reconstruction example**.
* All computation is performed inside Docker for maximum reproducibility.
* The code can be easily extended to support additional datasets or evaluation metrics.
