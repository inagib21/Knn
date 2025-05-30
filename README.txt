# KNN Project - MNIST Digit Recognizer

This project implements a k-Nearest Neighbors (KNN) classifier to recognize handwritten digits from the MNIST dataset. It includes various C++ implementations and optimizations to explore performance enhancements for the KNN algorithm.

## Features & Optimizations Implemented

1.  **Data Handling & Preprocessing:**
    *   Loads MNIST data from CSV files.
    *   Image preprocessing: Includes centering and cropping digits to a smaller target resolution (e.g., 20x20) to reduce dimensionality and improve focus on the digit itself.

2.  **Core KNN Classification Logic (Evolution):**
    *   **Initial Brute-Force:** Calculates distances to all training points, then sorts to find the k-nearest.
    *   **Optimized Brute-Force (Iterative Improvements):**
        *   **Insertion Sort based:** Maintained a sorted list of k-best neighbors by inserting new candidates, avoiding a full sort each time.
        *   **`std::nth_element`:** Used to find the k-th smallest distance in linear average time, then selecting the k smallest.
        *   **`std::priority_queue`:** Maintained a max-heap of size k to efficiently keep track of the k smallest distances found so far.
    *   **KD-Tree Implementation:**
        *   Constructs a KD-Tree from the training data to enable faster nearest neighbor searches, significantly reducing the number of distance calculations compared to brute-force for large datasets.
        *   Supports leaf nodes: When a node in the tree construction phase has fewer points than a specified `LEAF_SIZE`, it becomes a leaf node storing all those points directly. This is a common optimization in production KD-Trees (similar to scikit-learn's approach).

3.  **Performance Enhancements (Parallelism):**
    *   **Parallel K-Value Experimentation:** The `experimentWithKValues` function, which tests the classifier's performance across a range of `k` values, uses `std::async` to run evaluations for different `k` values concurrently.
    *   **Parallel Test Sample Evaluation:** Within `evaluateAccuracy` and `getConfusionMatrix`, the classification of individual test samples is parallelized using `std::async`. This speeds up the evaluation phase for a single `k` value, especially with many test images.

4.  **Evaluation & Analysis:**
    *   Calculates accuracy.
    *   Generates and prints confusion matrices.
    *   Provides per-class accuracy details.

## Requirements
- CMake (https://cmake.org/download/)
- C++14 compliant compiler (or newer, due to use of initialized lambda captures). Examples: MinGW-w64 for Windows, g++/clang for Mac/Linux, or Visual Studio.
- MNISTCSV/ data folder (containing `mnist_train.csv` and `mnist_test.csv`).

## Build Steps (Cross-Platform)

### 1. Ensure you have CMake and a C++14 (or newer) compiler installed.

### 2. Clone the repository or unzip the project and open a terminal/command prompt in the project folder (`HonorsProject`).

### 3. Run the following commands:
```
mkdir build
cd build
cmake ..
cmake --build .
```
- For Visual Studio users, you might want to specify a generator: `cmake .. -G "Visual Studio 17 2022"` (adjust version as needed), then build using Visual Studio or `cmake --build .`.
- This will create an executable (e.g., `knn.exe` on Windows, `knn` on Mac/Linux) in the `build` directory.

### 4. Ensure the `MNISTCSV/` data folder is located in the `HonorsProject` directory (i.e., one level up from where the `build/knn` executable is, so the program can find `../MNISTCSV/mnist_train.csv`).

## Run
- Navigate to the `HonorsProject` directory (if you are in `build`, `cd ..`).
- Execute from the `HonorsProject` directory:
    - On Windows: `build\knn.exe`
    - On Mac/Linux: `./build/knn`

## Notes
- The program expects the `MNISTCSV` directory to be at `../MNISTCSV/` relative to the executable's location if run from the `build` directory, or `./MNISTCSV/` if run from the `HonorsProject` directory after adjusting paths or if the executable is copied there. The current code in `main.cpp` uses paths like `"MNISTCSV/mnist_train.csv"`, implying the executable is run from the `HonorsProject` root, or that `MNISTCSV` is in the same directory as the executable.
  For simplicity with the current build and run instructions, running `./build/knn` from the `HonorsProject` directory is the most straightforward way to ensure data paths are resolved correctly as `MNISTCSV/mnist_train.csv` etc.
- If you encounter issues with file paths, ensure the program can locate the CSV files.
- Do not share the `build/` folder or any Mac/Linux executables; let each user build on their own system.

## Notes
- If using Visual Studio, you can open the folder as a CMake project and build/run from the IDE.
- Do not share the `build/` folder or any Mac/Linux executables; let each user build on their own system.
- If you encounter issues with file paths, use forward slashes (`/`) or double backslashes (`\\`) in your code for cross-platform compatibility. 