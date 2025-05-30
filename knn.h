#ifndef KNN_H
#define KNN_H

#include <vector>
#include <string>
#include <chrono>
#include <limits> // Required for std::numeric_limits
#include <queue>  // Added for std::priority_queue

struct DataPoint {
    std::vector<double> features;
    int label;
    size_t original_index; // Useful for KD-Tree if we sort indices
};

// Forward declaration
class KNNClassifier;

struct KDNode {
    // For internal nodes, point_index refers to the splitting point in training_data.
    // For leaf nodes, this might be unused or refer to the first point.
    size_t point_index; 

    KDNode* left = nullptr;
    KDNode* right = nullptr;
    int split_dimension = -1; // Dimension used for splitting at this node
    double split_value = 0.0;   // The value at which the split occurs

    bool is_leaf = false;                      // True if this is a leaf node
    std::vector<size_t> point_indices_in_leaf; // Stores indices of points if it's a leaf

    // Constructor for internal nodes (or potentially single-point leaves if LEAF_SIZE is 1)
    KDNode(size_t idx) : point_index(idx) {}

    // Constructor for leaf nodes (can be adapted or a new one added)
    // KDNode() : is_leaf(true) {} // Example if we need a dedicated leaf constructor
};

class KNNClassifier {
private:
    std::vector<DataPoint> training_data; // Original training data
    int k;
    KDNode* tree_root = nullptr; // Root of the KD-Tree
    const int LEAF_SIZE = 10; // Example: Max points in a leaf node, tune as needed

    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) const;
    
    // KD-Tree helper methods
    KDNode* buildKDTreeRecursive(std::vector<size_t>& point_indices, int depth);
    void searchKDTreeRecursive(KDNode* current_node, const DataPoint& query_point, 
                               std::priority_queue<std::pair<double, int>>& k_nearest_neighbors, 
                               int depth) const;
    void deleteKDTreeRecursive(KDNode* node);

public:
    KNNClassifier(int k = 3);
    KNNClassifier(const KNNClassifier& other); // Copy constructor
    ~KNNClassifier(); // Destructor to clean up KD-Tree

    // Load data
    void loadTrainingData(const std::string& csv_filename);
    std::vector<DataPoint> loadTestData(const std::string& csv_filename); // Might not need modification if it just returns DataPoints
    
    // Classification
    int classify(const std::vector<double>& input_features) const; // Will use the KD-Tree
    
    // Evaluation
    double evaluateAccuracy(const std::vector<DataPoint>& test_data, int max_test_images = 1000) const;
    std::vector<std::vector<int>> getConfusionMatrix(const std::vector<DataPoint>& test_data, int max_test_images = 1000) const;
    
    // Analysis
    void printDatasetInfo() const;
    void experimentWithKValues(const std::vector<DataPoint>& test_data, const std::vector<int>& k_values, int max_test_images = 1000) const;
    
    // Getters/Setters
    void setK(int new_k);
    int getK() const;
};

#endif 