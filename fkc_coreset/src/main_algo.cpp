/**

    Greedy Set Cover Algorithm for FKC Coreset
    - read posting lists:
        - postgress
        - text file 
    - read labels file:
        - text file read 
    - initialize trackers:
        - coverage
        - fairness
    - while loop:
        - matrix multiplication optimizations
        - numpy?

*/

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <cmath>
#include <ctime>
#include <climits>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <map>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <functional>
using namespace std;


bool check_for_zeros(vector<int> v) {
    bool zeros = std::all_of(v.begin(), v.end(), [](int i){ return i == 0;});
    return zeros;
}


/***
 * GFKC Algorithm : computes the FKC coreset for the dataset
 *                  based on the constraints
 * 
 * @param dataset name of the dataset
 * @param coverage_factor coverage factor constaint
 * @param distribution_req class based fairness constraint
 * @param num_classes number of classes
 * @param dataset_size size of the dataset
*/

void gfkc(string dataset, double coverage_factor, int distribution_req, int num_classes, int dataset_size) {
    // Reading posting list 
    // string posting_list_file = "/localdisk3/data-selection/data/metadata/" + dataset + "/" + to_string(coverage_factor) + "/resnet-18.txt";
    // cout << posting_list_file << endl;
    double pre_time = 0.0;
    double algo_time = 0.0;
    string posting_list_file = "/localdisk3/data-selection/data/metadata/cifar10/0.9/resnet-18_rf.txt";
    std::chrono::time_point<std::chrono::high_resolution_clock> pre_start, pre_end;
    std::chrono::duration<double> pre_elapsed;
    pre_start = std::chrono::high_resolution_clock::now();
    std::ifstream infile(posting_list_file);
    map<int, vector<int>> posting_list;
    if (infile.is_open()) {
        string line;
        int key = 0;
        while (getline(infile, line)) {
            // cout << line << endl;
            // break;
            stringstream ss(line);
            string value;
            vector<int> values(dataset_size, 0);
            while (ss >> value) {
                // cout << value << endl;
                values[stoi(value)] = 1;
            }

            posting_list[key] = values;
            key += 1;
        }
        

    } else {
        cout << "Error opening input file: " << posting_list_file << endl;
    }
    infile.close();
    string labels_file = "/localdisk3/data-selection/data/metadata/cifar10/labels.txt";
    map<int, vector<int>> labels_dict;
    std::ifstream lfile(labels_file);
    if (lfile.is_open()) {
        string line;
        while (getline(lfile, line)) {
            stringstream ss(line);
            string value;
            char del = ':';
            int iter = 0;
            int id;
            int label;
            while (!ss.eof()) {
                getline(ss, value, del);
                if (iter == 0) {
                    id = stoi(value);
                    iter += 1;
                } else {
                    label = stoi(value);
                }
            }
            vector<int> label_array(num_classes, 0);
            label_array[label] = 1;
            labels_dict[id] = label_array;
        }
    } else {
        cout << "Error opening input file: " << labels_file << endl;
    }
    lfile.close();
    // cout << to_string(labels_dict[50000]) << endl;
    pre_end = std::chrono::high_resolution_clock::now();
    pre_elapsed = pre_end - pre_start;
    pre_time = pre_elapsed.count();
    cout << "Preprocessing Time: " << to_string(pre_time) << endl;

    // initialize trackers
    /**
     * @todo coverage tracker sparse points
     * @todo fairness tracker class wise distribution req
    */
    vector<int> coverage_tracker(dataset_size, coverage_factor);
    vector<int> fairness_tracker(num_classes, distribution_req);
    set<int> coreset;
    set<int> delta;
    for (int i = 0; i < dataset_size; i++) {
        delta.insert(i);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> algo_start, algo_end;
    std::chrono::duration<double> algo_elapsed;
    algo_start = std::chrono::high_resolution_clock::now();
    while (((!check_for_zeros(coverage_tracker)) || (!check_for_zeros(fairness_tracker))) && coreset.size() < dataset_size) {
        vector<int> possible_candidates;
        set_difference(delta.begin(), delta.end(), coreset.begin(), coreset.end(), inserter(possible_candidates, possible_candidates.begin()));
        int best_point = -1;
        int max_score = INT_MIN;
        for (int i : possible_candidates) {
            // vector<int> coverage_compute;
            // vector<int> fairness_compute;
            // transform(posting_list[i].begin(), posting_list[i].end(), coverage_tracker.begin(), coverage_compute.begin(), multiplies<int>());
            // transform(labels_dict[i].begin(), labels_dict[i].end(), fairness_tracker.begin(), fairness_compute.begin(), multiplies<int>());
            // int coverage_score = accumulate(coverage_compute.begin(), coverage_compute.end(), 0);
            // int fairness_score = accumulate(fairness_compute.begin(), fairness_compute.end(), 0);
            int coverage_score = inner_product(posting_list[i].begin(), posting_list[i].end(), coverage_tracker.begin(), 0);
            int fairness_score = inner_product(labels_dict[i].begin(), labels_dict[i].end(), fairness_tracker.begin(), 0);
            // cout << "Coverage score is " << coverage_score << endl;
            // cout << "Fairness score is " << fairness_score << endl;
            if ((coverage_score + fairness_score) > max_score) {
                best_point = i;
                max_score = coverage_score + fairness_score;
            }
        }

        if (best_point != -1) {
            cout << "Found a point" << endl;
            coreset.insert(best_point);
            cout << "Coreset size: " << to_string(coreset.size()) << endl;
            // vector<int> temp_coverage_tracker;
            // vector<int> temp_fairness_tracker;
            // transform(posting_list[best_point].begin(), posting_list[best_point].end(), coverage_tracker.begin(), temp_coverage_tracker.begin(), minus<int>());
            // transform(labels_dict[best_point].begin(), labels_dict[best_point].end(), fairness_tracker.begin(), temp_fairness_tracker.begin(), minus<int>());
            // coverage_tracker = temp_coverage_tracker;
            // fairness_tracker = temp_fairness_tracker;
            vector<int> pl_best = posting_list[best_point];
            for (auto i : pl_best) {
                // cout << i << endl;
                if (i == 1) {
                    // cout << "here" << endl;
                    // cout << coverage_tracker[i] << endl;
                    // coverage_tracker[i] -= 1;
                    // cout << coverage_tracker[i] << endl;
                    if (coverage_tracker[i] != 0) {
                        coverage_tracker[i] -= 1;
                    }
                    // cout << coverage_tracker[i] << endl;
                    // break;
                }
            }
            vector<int> labels_best = labels_dict[best_point];
            for (auto i : labels_best) {
                if (i == 1) {
                    // fairness_tracker[i] -= 1;
                    if (fairness_tracker[i] != 0) {
                        fairness_tracker[i] -= 1;
                    }
                }
            }
        } else {
            cout << "Cannot find a point" << endl;
            break;
        }

        if ((check_for_zeros(coverage_tracker)) && (check_for_zeros(fairness_tracker))) {
            cout << "Satisfied" << endl;
        } else {
            cout << "not satisfied" << endl;
        }
        
    }
    algo_end = std::chrono::high_resolution_clock::now();
    algo_elapsed = algo_end - algo_start;
    algo_time = algo_elapsed.count();

    cout << "Algorithm time: " << to_string(algo_time) << endl;
    
}













/**
 * 
 * Main Driver Function
*/

int main(int argc, char const *argv[]) {
    string dataset = argv[1];
    double coverage_factor = stod(argv[2]);
    int distribution_req = stoi(argv[3]);
    int num_classes = 10;
    int dataset_size = 50000;
    // cout << coverage_factor << endl;

    gfkc(dataset, coverage_factor, distribution_req, num_classes, dataset_size);
    return 0;
}