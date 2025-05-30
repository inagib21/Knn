# KNN Project - MNIST Digit Recognizer ✏️🔢

This project implements a k-Nearest Neighbors (KNN) classifier to recognize handwritten digits from the MNIST dataset. It includes various C++ implementations and optimizations to explore performance enhancements for the KNN algorithm.

## ✨ Features & Optimizations Implemented

1.  **Data Handling & Preprocessing: 🗂️🖼️**
    *   Loads MNIST data from CSV files.
    *   Image preprocessing: Includes centering and cropping digits to a smaller target resolution (e.g., 20x20) to reduce dimensionality and improve focus on the digit itself.

2.  **Core KNN Classification Logic (Evolution): 🧠💡**
    *   **Initial Brute-Force:** Calculates distances to all training points, then sorts to find the k-nearest.
    *   **Optimized Brute-Force (Iterative Improvements):**
        *   **Insertion Sort based:** Maintained a sorted list of k-best neighbors by inserting new candidates, avoiding a full sort each time.
        *   **`std::nth_element`:** Used to find the k-th smallest distance in linear average time, then selecting the k smallest.
        *   **`std::priority_queue`:** Maintained a max-heap of size k to efficiently keep track of the k smallest distances found so far.
    *   **KD-Tree Implementation:** 🌳
        *   Constructs a KD-Tree from the training data to enable faster nearest neighbor searches, significantly reducing the number of distance calculations compared to brute-force for large datasets.
        *   Supports **leaf nodes:** When a node in the tree construction phase has fewer points than a specified `LEAF_SIZE`, it becomes a leaf node storing all those points directly. This is a common optimization in production KD-Trees (similar to scikit-learn's approach).

3.  **Performance Enhancements (Parallelism): 🚀⚡**
    *   **Parallel K-Value Experimentation:** The `experimentWithKValues` function, which tests the classifier's performance across a range of `k` values, uses `std::async` to run evaluations for different `k` values concurrently.
    *   **Parallel Test Sample Evaluation:** Within `evaluateAccuracy` and `getConfusionMatrix`, the classification of individual test samples is parallelized using `std::async`. This speeds up the evaluation phase for a single `k` value, especially with many test images.

4.  **Evaluation & Analysis: 📊📈**
    *   Calculates accuracy.
    *   Generates and prints confusion matrices.
    *   Provides per-class accuracy details.

## 📋 Requirements
- CMake (https://cmake.org/download/)
- C++14 compliant compiler (or newer, due to use of initialized lambda captures). Examples: MinGW-w64 for Windows, g++/clang for Mac/Linux, or Visual Studio.
- `MNISTCSV/` data folder (containing `mnist_train.csv` and `mnist_test.csv`). Ensure Git LFS is used for these large files.

## 🛠️ Build Steps (Cross-Platform)

### 1. Ensure you have CMake and a C++14 (or newer) compiler installed.

### 2. Clone the repository. If you haven't already, ensure Git LFS is installed (`brew install git-lfs`, `git lfs install`). Then pull LFS files:
```bash
git clone https://github.com/inagib21/Knn.git
cd Knn # Or your project directory name
git lfs pull
```

### 3. Open a terminal/command prompt in the project folder (e.g., `HonorsProject` or `Knn`).

### 4. Run the following commands to build:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```
- For Visual Studio users, you might want to specify a generator: `cmake .. -G "Visual Studio 17 2022"` (adjust version as needed), then build using Visual Studio or `cmake --build .`.
- This will create an executable (e.g., `knn.exe` on Windows, `knn` on Mac/Linux) in the `build` directory.

### 5. Ensure the `MNISTCSV/` data folder is located in the project root directory (e.g., `HonorsProject` or `Knn`), so the program can find `MNISTCSV/mnist_train.csv` when run from the project root.

## ▶️ Run
- Navigate to the project root directory (e.g., `HonorsProject` or `Knn`).
- Execute from the project root directory:
    - On Windows: `build\knn.exe`
    - On Mac/Linux: `./build/knn`

## 📝 Notes
- The program is designed to be run from the project root directory so that relative paths to data like `MNISTCSV/mnist_train.csv` resolve correctly.
- If using Visual Studio, you can often open the folder as a CMake project and build/run from the IDE, but ensure the working directory for execution is set to the project root for data file access.
- Do not share the `build/` folder or any Mac/Linux executables; let each user build on their own system.
- If you encounter issues with file paths, double-check the execution directory and the relative paths used in the code (`main.cpp`).
- Consider adding a `.gitignore` file to exclude the `build/` directory and other non-source files (like `.DS_Store`) from version control.

## Notes
- If using Visual Studio, you can open the folder as a CMake project and build/run from the IDE.
- Do not share the `build/` folder or any Mac/Linux executables; let each user build on their own system.
- If you encounter issues with file paths, use forward slashes (`/`) or double backslashes (`\\`) in your code for cross-platform compatibility. 