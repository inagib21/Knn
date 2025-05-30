#include "knn.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

void printUsage(const char* programName) {
    cout << "Usage: " << programName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --train-csv FILE       Path to training CSV file (default: MNISTCSV/mnist_train.csv)" << endl;
    cout << "  --test-csv FILE        Path to test CSV file (default: MNISTCSV/mnist_test.csv)" << endl;
    cout << "  --k-values K1,K2,...   Comma-separated list of k values to test (default: 3,5,7,9,11)" << endl;
    cout << "  --max-test N           Maximum number of test images to use (default: 1000)" << endl;
    cout << "  --help                 Show this help message" << endl;
}

int main(int argc, char* argv[]) {
    // Default values for CSV data
    string train_csv = "MNISTCSV/mnist_train.csv";
    string test_csv = "MNISTCSV/mnist_test.csv";
    vector<int> k_values = {3, 5, 7, 9, 11};
    int max_test = 1000;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--train-csv" && i + 1 < argc) {
            train_csv = argv[++i];
        } else if (arg == "--test-csv" && i + 1 < argc) {
            test_csv = argv[++i];
        } else if (arg == "--k-values" && i + 1 < argc) {
            k_values.clear();
            string k_str = argv[++i];
            size_t pos = 0;
            while ((pos = k_str.find(',')) != string::npos) {
                k_values.push_back(stoi(k_str.substr(0, pos)));
                k_str.erase(0, pos + 1);
            }
            if (!k_str.empty()) {
                k_values.push_back(stoi(k_str));
            }
        } else if (arg == "--max-test" && i + 1 < argc) {
            max_test = stoi(argv[++i]);
        } else {
            // Handle potential old flags to avoid confusion, or just report unknown
            if (arg == "--train-images" || arg == "--train-labels" || arg == "--test-images" || arg == "--test-labels") {
                 cerr << "Error: Obsolete flag detected. Use --train-csv and --test-csv for CSV data." << endl;
            } else {
                cerr << "Unknown option: " << arg << endl;
            }
            printUsage(argv[0]);
            return 1;
        }
    }

    try {
        // Create and initialize the KNN classifier with a default k=3 (can be changed by experiment)
        KNNClassifier knn(3);

        // Load training data from CSV
        cout << "Loading training data..." << endl;
        knn.loadTrainingData(train_csv);
        knn.printDatasetInfo();

        // Load test data from CSV
        cout << "\nLoading test data..." << endl;
        vector<DataPoint> test_data = knn.loadTestData(test_csv);

        // Experiment with different k values
        knn.experimentWithKValues(test_data, k_values, max_test);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
} 