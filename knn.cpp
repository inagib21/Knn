#include "knn.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <string>
#include <future>
#include <thread>
#include <queue>
#include <numeric>

using namespace std;
using namespace std::chrono;

KNNClassifier::KNNClassifier(int k) : k(k), tree_root(nullptr) {}

// Copy constructor: Does not deep copy the tree to avoid double free in current usage.
// Copied instances will not use the KD-Tree unless it's rebuilt for them.
KNNClassifier::KNNClassifier(const KNNClassifier& other) 
    : training_data(other.training_data), // This is a deep copy of DataPoints
      k(other.k),
      tree_root(nullptr), // CRITICAL: Copied instance does not own/share the original tree by default
      LEAF_SIZE(other.LEAF_SIZE)
{
    // If a deep copy of the KD-Tree were desired, it would be done here.
    // For now, the copy gets no tree to prevent double free issues with current experimentWithKValues.
    // This means the copied instances in experimentWithKValues will currently not benefit from the KD-Tree speedup.
    // To make them benefit, experimentWithKValues would need to call buildKDTreeRecursive for the copy,
    // or a full deep copy of the tree would be needed.
}

KNNClassifier::~KNNClassifier() {
    deleteKDTreeRecursive(tree_root);
}

void KNNClassifier::deleteKDTreeRecursive(KDNode* node) {
    if (node == nullptr) {
        return;
    }
    deleteKDTreeRecursive(node->left);
    deleteKDTreeRecursive(node->right);
    delete node;
}

void KNNClassifier::setK(int new_k) {
    if (new_k <= 0) {
        throw invalid_argument("k must be positive");
    }
    k = new_k;
}

int KNNClassifier::getK() const {
    return k;
}

double KNNClassifier::euclideanDistance(const vector<double>& a, const vector<double>& b) const {
    if (a.size() != b.size()) {
        throw invalid_argument("Vectors must have the same size");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

// Helper function (can be private or outside the class)
vector<double> preprocessImage(const vector<double>& raw_features, int original_width, int original_height, int target_width, int target_height) {
    // Find bounding box
    int min_row = original_height, max_row = -1, min_col = original_width, max_col = -1;
    for (int r = 0; r < original_height; ++r) {
        for (int c = 0; c < original_width; ++c) {
            // Check for non-zero pixel (allowing for small floating point values)
            if (raw_features[r * original_width + c] > 1e-6) { 
                if (r < min_row) min_row = r;
                if (r > max_row) max_row = r;
                if (c < min_col) min_col = c;
                if (c > max_col) max_col = c;
            }
        }
    }

    // Handle empty images (all zeros) - return a zero vector
    if (max_row < min_row || max_col < min_col) {
        return vector<double>(target_width * target_height, 0.0);
    }

    int cropped_height = max_row - min_row + 1;
    int cropped_width = max_col - min_col + 1;

    // Create target canvas (initialized to 0.0)
    vector<double> processed_features(target_width * target_height, 0.0);

    // Calculate padding offsets to center the cropped image
    int row_offset = (target_height - cropped_height) / 2;
    int col_offset = (target_width - cropped_width) / 2;

    // Copy cropped image to the center of the target canvas
    for (int r = 0; r < cropped_height; ++r) {
        for (int c = 0; c < cropped_width; ++c) {
            int target_r = row_offset + r;
            int target_c = col_offset + c;

            // Ensure we stay within target bounds (important if cropped > target)
            if (target_r >= 0 && target_r < target_height && target_c >= 0 && target_c < target_width) {
                 // Get pixel from original raw_features within the bounding box
                processed_features[target_r * target_width + target_c] = 
                    raw_features[(min_row + r) * original_width + (min_col + c)];
            }
        }
    }

    return processed_features;
}

void KNNClassifier::loadTrainingData(const string& csv_filename) {
    const int ORIGINAL_WIDTH = 28;
    const int ORIGINAL_HEIGHT = 28;
    const int TARGET_WIDTH = 20; 
    const int TARGET_HEIGHT = 20;

    ifstream file(csv_filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open CSV file: " + csv_filename);
    }

    // Clear existing data and tree
    training_data.clear();
    deleteKDTreeRecursive(tree_root);
    tree_root = nullptr;

    string line;
    int line_count = 0;
    size_t data_idx_counter = 0; // For original_index

    // Read header line (and discard it)
    if (!getline(file, line)) {
         throw runtime_error("Error reading header or empty file: " + csv_filename);
    }
    line_count++;

    cout << "Loading and preprocessing training data from " << csv_filename << "..." << endl;

    while (getline(file, line)) {
        line_count++;
        stringstream ss(line);
        string cell;
        // vector<double> features; // Not needed here directly
        int label = -1;
        int cell_count = 0;

        // Read label (first cell)
        if (getline(ss, cell, ',')) {
            try {
                label = stoi(cell);
            } catch (const std::invalid_argument& ia) {
                cerr << "Warning: Invalid label format at line " << line_count << ": '" << cell << "'. Skipping row." << endl;
                continue; 
            } catch (const std::out_of_range& oor) {
                 cerr << "Warning: Label out of range at line " << line_count << ": '" << cell << "'. Skipping row." << endl;
                 continue; 
            }
            cell_count++;
        } else {
             cerr << "Warning: Empty or incomplete line " << line_count << ". Skipping row." << endl;
             continue; // Skip this row
        }

        vector<double> raw_features;
        raw_features.reserve(ORIGINAL_WIDTH * ORIGINAL_HEIGHT); 
        while (getline(ss, cell, ',')) {
             try {
                raw_features.push_back(stod(cell) / 255.0); 
            } catch (const std::invalid_argument& ia) {
                 cerr << "Warning: Invalid feature format at line " << line_count << ", cell " << cell_count + 1 << ": '" << cell << "'. Treating as 0.0." << endl;
                 raw_features.push_back(0.0); 
            } catch (const std::out_of_range& oor) {
                 cerr << "Warning: Feature value out of range at line " << line_count << ", cell " << cell_count + 1 << ": '" << cell << "'. Treating as 0.0." << endl;
                 raw_features.push_back(0.0);
            }
            cell_count++;
        }
        
        if (raw_features.size() != ORIGINAL_WIDTH * ORIGINAL_HEIGHT) {
             cerr << "Warning: Incorrect number of features at line " << line_count
                  << ". Expected " << ORIGINAL_WIDTH * ORIGINAL_HEIGHT << ", found " << raw_features.size()
                  << ". Skipping row." << endl;
             continue;       
        }

        if (label != -1) {
             vector<double> processed_features = preprocessImage(raw_features, ORIGINAL_WIDTH, ORIGINAL_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT);
             training_data.push_back({processed_features, label, data_idx_counter++});
        }

        if (training_data.size() % 5000 == 0 && training_data.size() > 0) {
            cout << "Loaded and processed " << training_data.size() << " training samples..." << endl;
        }
    }

    if (training_data.empty()) {
         throw runtime_error("No valid data loaded from " + csv_filename);
    }

    cout << "Finished loading and preprocessing training dataset. Total samples: " << training_data.size() << endl;
    if (!training_data.empty()) {
         cout << "New feature vector size: " << training_data[0].features.size() << endl;
    }

    // Build KD-Tree
    if (!training_data.empty()) {
        cout << "Building KD-Tree..." << endl;
        std::vector<size_t> point_indices(training_data.size());
        std::iota(point_indices.begin(), point_indices.end(), 0); // Fill with 0, 1, 2, ...
        tree_root = buildKDTreeRecursive(point_indices, 0);
        cout << "KD-Tree built." << endl;
    }
}

// KD-Tree helper methods implementation
KDNode* KNNClassifier::buildKDTreeRecursive(std::vector<size_t>& point_indices, int depth) {
    if (point_indices.empty()) {
        return nullptr;
    }

    // If the number of points is below or equal to LEAF_SIZE, create a leaf node
    if (point_indices.size() <= LEAF_SIZE) {
        // For a leaf node, we might not need a specific splitting point_index from the subset,
        // or we could use the first one. Let's use 0 for simplicity as it won't be a split point.
        // The crucial part is storing all indices in point_indices_in_leaf.
        KDNode* leaf_node = new KDNode(0); // point_index is somewhat arbitrary for a leaf holder
        leaf_node->is_leaf = true;
        leaf_node->point_indices_in_leaf = point_indices; // Store all indices
        leaf_node->left = nullptr; // No children for leaf nodes
        leaf_node->right = nullptr;
        leaf_node->split_dimension = -1; // No split dimension for leaves
        return leaf_node;
    }

    // Select axis based on depth
    // Assumes all features vectors in training_data have the same size.
    // And training_data[0] exists (checked before calling buildKDTreeRecursive).
    int num_dimensions = training_data[0].features.size();
    if (num_dimensions == 0) { // Should not happen with valid data
        throw std::runtime_error("Data points have 0 dimensions, cannot build KD-Tree.");
    }
    int axis = depth % num_dimensions;

    // Sort point_indices based on the chosen axis
    // This sort modifies the point_indices vector in place for the current recursive call's scope.
    std::sort(point_indices.begin(), point_indices.end(), 
              [&](size_t idx1, size_t idx2) {
                  return training_data[idx1].features[axis] < training_data[idx2].features[axis];
              });

    // Choose median as pivot
    size_t median_idx_in_subset = point_indices.size() / 2;
    size_t median_original_data_index = point_indices[median_idx_in_subset];

    KDNode* node = new KDNode(median_original_data_index); // Node stores index to original training_data
    node->split_dimension = axis;
    node->split_value = training_data[median_original_data_index].features[axis];
    node->is_leaf = false; // This is an internal node

    // Recursively build left and right subtrees
    // Create new vectors for left and right children's indices
    std::vector<size_t> left_indices(point_indices.begin(), point_indices.begin() + median_idx_in_subset);
    std::vector<size_t> right_indices(point_indices.begin() + median_idx_in_subset + 1, point_indices.end());

    node->left = buildKDTreeRecursive(left_indices, depth + 1);
    node->right = buildKDTreeRecursive(right_indices, depth + 1);

    return node;
}

vector<DataPoint> KNNClassifier::loadTestData(const string& csv_filename) {
    const int ORIGINAL_WIDTH = 28;
    const int ORIGINAL_HEIGHT = 28;
    const int TARGET_WIDTH = 20; 
    const int TARGET_HEIGHT = 20;

    vector<DataPoint> test_data;
    ifstream file(csv_filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open test CSV file: " + csv_filename);
    }

    string line;
    int line_count = 0;

     // Read header line (and discard it)
    if (!getline(file, line)) {
         throw runtime_error("Error reading header or empty file: " + csv_filename);
    }
    line_count++;

    cout << "Loading and preprocessing test data from " << csv_filename << "..." << endl;

    while (getline(file, line)) {
        line_count++;
        stringstream ss(line);
        string cell;
        vector<double> features;
        int label = -1;
        int cell_count = 0;

        // Read label (first cell)
         if (getline(ss, cell, ',')) {
            try {
                label = stoi(cell);
            } catch (const std::invalid_argument& ia) {
                cerr << "Warning: Invalid label format at line " << line_count << ": '" << cell << "'. Skipping row." << endl;
                continue; 
            } catch (const std::out_of_range& oor) {
                 cerr << "Warning: Label out of range at line " << line_count << ": '" << cell << "'. Skipping row." << endl;
                 continue; 
            }
            cell_count++;
        } else {
             cerr << "Warning: Empty or incomplete line " << line_count << ". Skipping row." << endl;
             continue; 
        }

        // Read features (remaining cells) into a temporary raw vector
        vector<double> raw_features;
        raw_features.reserve(ORIGINAL_WIDTH * ORIGINAL_HEIGHT); 
        while (getline(ss, cell, ',')) {
             try {
                raw_features.push_back(stod(cell) / 255.0);
            } catch (const std::invalid_argument& ia) {
                 cerr << "Warning: Invalid feature format at line " << line_count << ", cell " << cell_count + 1 << ": '" << cell << "'. Treating as 0.0." << endl;
                 raw_features.push_back(0.0);
            } catch (const std::out_of_range& oor) {
                 cerr << "Warning: Feature value out of range at line " << line_count << ", cell " << cell_count + 1 << ": '" << cell << "'. Treating as 0.0." << endl;
                 raw_features.push_back(0.0);
            }
            cell_count++;
        }
        
        // Ensure correct number of features
        if (raw_features.size() != ORIGINAL_WIDTH * ORIGINAL_HEIGHT) {
             cerr << "Warning: Incorrect number of features in test data at line " << line_count
                  << ". Expected " << ORIGINAL_WIDTH * ORIGINAL_HEIGHT << ", found " << raw_features.size()
                  << ". Skipping row." << endl;
             continue;       
        }

        if (label != -1) {
            // Preprocess the image
            vector<double> processed_features = preprocessImage(raw_features, ORIGINAL_WIDTH, ORIGINAL_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT);
            
            // Add the processed features and label to test data
            test_data.push_back({processed_features, label});
        }

        if (test_data.size() % 1000 == 0 && test_data.size() > 0) {
            cout << "Loaded and processed " << test_data.size() << " test samples..." << endl;
        }
    }

     if (test_data.empty()) {
         throw runtime_error("No valid data loaded from " + csv_filename);
    }

    cout << "Finished loading and preprocessing test dataset. Total samples: " << test_data.size() << endl;
    if (!test_data.empty()) {
        cout << "New feature vector size: " << test_data[0].features.size() << endl;
    }
    return test_data;
}

// New classify method using KD-Tree, with fallback to brute-force if tree is not available
int KNNClassifier::classify(const std::vector<double>& input_features) const {
    if (training_data.empty()) {
        throw std::runtime_error("No training data available");
    }
    if (input_features.size() != training_data[0].features.size()) {
        throw std::invalid_argument("Input vector size does not match training data feature size");
    }

    // Max-priority queue for {distance, label} or {squared_distance, label}
    std::priority_queue<std::pair<double, int>> k_nearest_neighbors_pq;

    if (tree_root != nullptr) {
        // Use KD-Tree search
        DataPoint query_point = {input_features, -1, 0};
        searchKDTreeRecursive(tree_root, query_point, k_nearest_neighbors_pq, 0);
    } else {
        // Fallback to brute-force using priority queue (like before KD-Tree)
        // Ensure this part uses squared distances if searchKDTreeRecursive does, for consistency, or handle it.
        // The searchKDTreeRecursive currently puts squared distances in the PQ.
        for (const auto& point : training_data) {
            double dist_sq = 0.0;
            for (size_t i = 0; i < input_features.size(); ++i) {
                double diff = input_features[i] - point.features[i];
                dist_sq += diff * diff;
            }

            if (k_nearest_neighbors_pq.size() < k) {
                k_nearest_neighbors_pq.push({dist_sq, point.label});
            } else if (dist_sq < k_nearest_neighbors_pq.top().first) {
                k_nearest_neighbors_pq.pop();
                k_nearest_neighbors_pq.push({dist_sq, point.label});
            }
        }
    }

    // Extract labels and count votes
    std::vector<int> count(10, 0); // Assuming 10 classes
    if (k_nearest_neighbors_pq.empty()){
        if (!training_data.empty()) { 
             return training_data[0].label; 
        } else {
            throw std::runtime_error("No neighbors found and no training data to default to.");
        }
    }

    while (!k_nearest_neighbors_pq.empty()) {
        int label = k_nearest_neighbors_pq.top().second;
        if (label >= 0 && label < 10) { 
            count[label]++;
        }
        k_nearest_neighbors_pq.pop();
    }
    
    return std::max_element(count.begin(), count.end()) - count.begin();
}

void KNNClassifier::searchKDTreeRecursive(KDNode* current_node, const DataPoint& query_point, 
                                       std::priority_queue<std::pair<double, int>>& k_nearest_neighbors, 
                                       int depth) const {
    if (current_node == nullptr) {
        return;
    }

    if (current_node->is_leaf) {
        // This is a leaf node, process all points in it
        for (size_t point_idx_in_leaf : current_node->point_indices_in_leaf) {
            const DataPoint& leaf_point = training_data[point_idx_in_leaf];
            double dist_sq = 0.0;
            for (size_t i = 0; i < query_point.features.size(); ++i) {
                double diff = query_point.features[i] - leaf_point.features[i];
                dist_sq += diff * diff;
            }

            if (k_nearest_neighbors.size() < k) {
                k_nearest_neighbors.push({dist_sq, leaf_point.label});
            } else if (dist_sq < k_nearest_neighbors.top().first) {
                k_nearest_neighbors.pop();
                k_nearest_neighbors.push({dist_sq, leaf_point.label});
            }
        }
        return; // No further recursion from a leaf
    }

    // --- This part is for internal nodes ---
    // 1. Calculate distance to current node's splitting point and update neighbors
    const DataPoint& node_point = training_data[current_node->point_index]; // This is the splitting point
    double dist_sq = 0.0; // Using squared Euclidean distance for internal comparisons to avoid sqrt
    for (size_t i = 0; i < query_point.features.size(); ++i) {
        double diff = query_point.features[i] - node_point.features[i];
        dist_sq += diff * diff;
    }
    // double actual_dist = std::sqrt(dist_sq); // Only needed if storing actual distance

    if (k_nearest_neighbors.size() < k) {
        k_nearest_neighbors.push({dist_sq, node_point.label}); // Store squared distance
    } else if (dist_sq < k_nearest_neighbors.top().first) {
        k_nearest_neighbors.pop();
        k_nearest_neighbors.push({dist_sq, node_point.label}); // Store squared distance
    }

    // 2. Determine which child to visit first
    int axis = current_node->split_dimension;
    double median_val = current_node->split_value; // training_data[current_node->point_index].features[axis];
    
    KDNode* first_child_to_visit = nullptr;
    KDNode* second_child_to_visit = nullptr;

    if (query_point.features[axis] < median_val) {
        first_child_to_visit = current_node->left;
        second_child_to_visit = current_node->right;
    } else {
        first_child_to_visit = current_node->right;
        second_child_to_visit = current_node->left;
    }

    // 3. Recursively search the first child (closer subtree)
    searchKDTreeRecursive(first_child_to_visit, query_point, k_nearest_neighbors, depth + 1);

    // 4. Check if the other side needs to be searched (pruning step)
    // If the k-th neighbor found so far is further than the distance from the query point
    // to the splitting plane, then the other subtree might contain closer points.
    double dist_to_plane_sq = query_point.features[axis] - median_val;
    dist_to_plane_sq *= dist_to_plane_sq;

    if (k_nearest_neighbors.size() < k || dist_to_plane_sq < k_nearest_neighbors.top().first) {
        searchKDTreeRecursive(second_child_to_visit, query_point, k_nearest_neighbors, depth + 1);
    }
}

double KNNClassifier::evaluateAccuracy(const vector<DataPoint>& test_data, int max_test_images) const {
    if (test_data.empty()) {
        throw invalid_argument("Test data is empty");
    }
    if (max_test_images <= 0) {
        throw invalid_argument("max_test_images must be positive");
    }

    int actual_max = min(max_test_images, static_cast<int>(test_data.size()));
    int correct_predictions = 0;

    auto start_time = high_resolution_clock::now();

    std::vector<std::future<int>> prediction_futures;
    prediction_futures.reserve(actual_max);

    // Launch asynchronous tasks for classifying each test image
    for (int i = 0; i < actual_max; ++i) {
        // Important: capture 'this' and test_data[i].features by value (or ensure lifetime)
        // 'this' is captured by value by default in a lambda if used.
        // test_data[i].features needs to be copied if test_data could go out of scope,
        // but here test_data is const& and should outlive the async calls within this function scope.
        prediction_futures.push_back(std::async(std::launch::async, 
            [this, features = test_data[i].features]() { // Capture features by copy
                return this->classify(features);
            }
        ));

        if ((i + 1) % 200 == 0) { // Optional: Log progress of task submission
            cout << "  Submitted " << (i + 1) << "/" << actual_max << " classification tasks for accuracy evaluation..." << endl;
        }
    }

    // Collect results and count correct predictions
    for (int i = 0; i < actual_max; ++i) {
        int predicted = prediction_futures[i].get(); // This will block until the future is ready
        if (predicted == test_data[i].label) {
            correct_predictions++;
        }
        if ((i + 1) % 100 == 0 && (i+1) >= actual_max) { // Log at the end or more frequently if desired
             cout << "  Collected " << (i + 1) << "/" << actual_max << " results for accuracy evaluation..." << endl;
        }
    }

    auto stop_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_time - start_time);

    double accuracy = static_cast<double>(correct_predictions) / actual_max * 100.0;
    cout << "Accuracy testing (k=" << this->k << ") completed in " << duration.count() << " ms for " << actual_max << " images." << endl;
    return accuracy;
}

vector<vector<int>> KNNClassifier::getConfusionMatrix(const vector<DataPoint>& test_data, int max_test_images) const {
    if (test_data.empty()) {
        throw invalid_argument("Test data is empty");
    }
    if (max_test_images <= 0) {
        throw invalid_argument("max_test_images must be positive");
    }

    int actual_max = min(max_test_images, static_cast<int>(test_data.size()));
    vector<vector<int>> confusion_matrix(10, vector<int>(10, 0));

    auto start_time = high_resolution_clock::now();

    struct PredictionResult {
        int actual_label;
        int predicted_label;
    };
    std::vector<std::future<PredictionResult>> prediction_futures;
    prediction_futures.reserve(actual_max);

    // Launch asynchronous tasks for classifying each test image
    for (int i = 0; i < actual_max; ++i) {
        prediction_futures.push_back(std::async(std::launch::async,
            [this, features = test_data[i].features, actual_label = test_data[i].label]() -> PredictionResult {
                int predicted = this->classify(features);
                return {actual_label, predicted};
            }
        ));
        if ((i + 1) % 200 == 0) { // Optional: Log progress of task submission
            cout << "  Submitted " << (i + 1) << "/" << actual_max << " classification tasks for confusion matrix..." << endl;
        }
    }

    // Collect results and populate confusion matrix
    for (int i = 0; i < actual_max; ++i) {
        PredictionResult result = prediction_futures[i].get();
        if (result.actual_label >= 0 && result.actual_label < 10 && 
            result.predicted_label >= 0 && result.predicted_label < 10) {
            confusion_matrix[result.actual_label][result.predicted_label]++;
        }
        if ((i + 1) % 100 == 0 && (i+1) >= actual_max) { // Log at the end or more frequently
            cout << "  Collected " << (i + 1) << "/" << actual_max << " results for confusion matrix..." << endl;
        }
    }
    
    auto stop_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_time - start_time);
    cout << "Confusion matrix (k=" << this->k << ") calculation completed in " << duration.count() << " ms for " << actual_max << " images." << endl;

    return confusion_matrix;
}

void KNNClassifier::printDatasetInfo() const {
    if (training_data.empty()) {
        cout << "Training dataset is empty." << endl;
        return;
    }

    cout << "\nTraining Dataset Statistics:" << endl;
    cout << "Total points: " << training_data.size() << endl;
    cout << "Feature vector size (after preprocessing): " << training_data[0].features.size() << endl;

    vector<int> labelCounts(10, 0);
    for (const auto& point : training_data) {
        if (point.label >= 0 && point.label < 10) {
            labelCounts[point.label]++;
        } else {
            cerr << "Warning: Found invalid label " << point.label << endl;
        }
    }

    cout << "\nPoints per digit:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "Digit " << i << ": " << labelCounts[i] << " points" << endl;
    }
}

void KNNClassifier::experimentWithKValues(const vector<DataPoint>& test_data, 
                                        const vector<int>& k_values, 
                                        int max_test_images) const {
    if (test_data.empty()) {
        throw invalid_argument("Test data is empty");
    }
    if (k_values.empty()) {
        throw invalid_argument("k_values vector is empty");
    }

    cout << "\n--- K-Value Experimentation ---" << endl;
    cout << "Testing with different k values on " << min(max_test_images, static_cast<int>(test_data.size())) 
         << " test images" << endl;

    // Track best performing k value
    double best_accuracy = -1.0;
    int best_k = -1;
    
    // Store futures for asynchronous tasks
    // Each task will return a pair: {accuracy, confusion_matrix}
    struct ExperimentResult {
        int k_val;
        double accuracy;
        vector<vector<int>> confusion_matrix;
    };
    vector<future<ExperimentResult>> futures;

    for (int k_value : k_values) {
        if (k_value <= 0) {
            cerr << "Warning: Skipping invalid k value: " << k_value << endl;
            continue;
        }

        // Launch asynchronous task for each k_value
        futures.push_back(async(launch::async, [this, test_data, k_value, max_test_images]() -> ExperimentResult {
            KNNClassifier temp_classifier = *this; // Create a copy for this thread
            temp_classifier.setK(k_value);

            cout << "\nStarting test for k = " << k_value << "..." << endl;
            double accuracy = temp_classifier.evaluateAccuracy(test_data, max_test_images);
            vector<vector<int>> confusion_matrix = temp_classifier.getConfusionMatrix(test_data, max_test_images);
            cout << "Finished test for k = " << k_value << ". Accuracy: " << fixed << setprecision(2) << accuracy << "%" << endl;
            return {k_value, accuracy, confusion_matrix};
        }));
    }

    // Collect results from futures
    vector<ExperimentResult> results;
    for (auto& fut : futures) {
        results.push_back(fut.get());
    }

    // Sort results by k_value for consistent output order (optional, but good for readability)
    sort(results.begin(), results.end(), [](const ExperimentResult& a, const ExperimentResult& b) {
        return a.k_val < b.k_val;
    });

    cout << "\n--- K-Value Experimentation Results ---" << endl;
    for (const auto& result : results) {
        cout << "\nResults for k = " << result.k_val << ":" << endl;
        cout << "Accuracy: " << fixed << setprecision(2) << result.accuracy << "%" << endl;

        // Update best k tracking
        if (result.accuracy > best_accuracy) {
            best_accuracy = result.accuracy;
            best_k = result.k_val;
        }

        cout << "\nConfusion Matrix (for k = " << result.k_val << "):";
        // Print confusion matrix header
        cout << "\n    ";
        for (int i = 0; i < 10; i++) cout << setw(4) << i;
        cout << "\n    " << string(40, '-') << endl;
        
        // Print confusion matrix rows
        for (int i = 0; i < 10; i++) {
            cout << i << " | ";
            for (int j = 0; j < 10; j++) {
                cout << setw(4) << result.confusion_matrix[i][j];
            }
            cout << endl;
        }

        // Print per-class accuracy for this k
        cout << "\nPer-class accuracy (for k = " << result.k_val << "):";
        int total_correct_cm = 0;
        int total_samples_cm = 0;
        for (int i = 0; i < 10; i++) {
            int total_in_class = 0;
            for (int j = 0; j < 10; j++) {
                total_in_class += result.confusion_matrix[i][j];
            }
            total_samples_cm += total_in_class;
            if (total_in_class > 0) {
                total_correct_cm += result.confusion_matrix[i][i];
                double class_accuracy = static_cast<double>(result.confusion_matrix[i][i]) / total_in_class * 100.0;
                cout << "\n  Class " << i << ": " << fixed << setprecision(2) << class_accuracy 
                     << "% (" << result.confusion_matrix[i][i] << "/" << total_in_class << ")";
            } else {
                cout << "\n  Class " << i << ": N/A (0 samples)";
            }
        }
        cout << "\nOverall from CM (k=" << result.k_val << "): " 
             << (total_samples_cm > 0 ? (static_cast<double>(total_correct_cm)/total_samples_cm*100.0) : 0.0) 
             << "%" << endl;
    }

    if (best_k != -1) {
        cout << "\nBest k value: " << best_k << " with accuracy: " << fixed << setprecision(2) << best_accuracy << "%" << endl;
    } else {
        cout << "\nNo valid k values were tested." << endl;
    }
    cout << "\n--- End of K-Value Experimentation ---" << endl;
}


    