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
#include <random>
#include <queue>
#include <omp.h>
using namespace std;


bool check_for_zeros(vector<int> v) {
    bool zeros = std::all_of(v.begin(), v.end(), [](int i){ return i == 0;});
    return zeros;
}

map<int, set<int>> get_posting_list(int datasetsize, string dataset_name) {

    string posting_list_file = "/localdisk3/data-selection/data/metadata/cifar10/0.9/resnet-18_rf.txt";
    std::ifstream infile(posting_list_file);
    map<int, set<int>> posting_list;
    if (infile.is_open()) {
        string line;
        int key = 0;
        while (getline(infile, line)) {
            stringstream ss(line);
            string value;
            set<int> values;
            while (ss >> value) {
                values.insert(stoi(value));
            }

            posting_list[key] = values;
            key += 1;
        }
        

    } else {
        cout << "Error opening input file: " << posting_list_file << endl;
    }
    infile.close();

    return posting_list;
    
}


map<int, vector<int>> get_posting_list_sc(int datasetsize, string dataset_name, set<int> candidates) {

    string posting_list_file = "/localdisk3/data-selection/data/metadata/cifar10/0.9/resnet-18_rf.txt";
    std::ifstream infile(posting_list_file);
    map<int, vector<int>> posting_list;
    if (infile.is_open()) {
        string line;
        int key = 0;
        while (getline(infile, line)) {
            stringstream ss(line);
            set<int>::iterator it = candidates.find(key);
            if (it != candidates.end()) {
                string value;
                vector<int> values(datasetsize, 0);
                while (ss >> value) {
                    values[stoi(value)] += 1;
                }

                posting_list[key] = values;    
            }
            key += 1;
        }
        

    } else {
        cout << "Error opening input file: " << posting_list_file << endl;
    }
    infile.close();

    return posting_list;
    
}


set<int> set_cover(vector<int> coverage_tracker, set<int> candidates, int dataset_size, string dataset_name) {
    set<int> solution;
    map<int, vector<int>> posting_list = get_posting_list_sc(dataset_size, dataset_name, candidates);
    while ((!check_for_zeros(coverage_tracker)) && solution.size() < candidates.size()) {
        vector<int> possible_candidates;
        set_difference(candidates.begin(), candidates.end(), solution.begin(), solution.end(), inserter(possible_candidates, possible_candidates.begin()));
        int best_point = -1;
        int max_score = INT_MIN;

        omp_lock_t sc_write_lock;
        omp_init_lock(&sc_write_lock);

        #pragma omp parallel for
        for (int j = 0; j < possible_candidates.size(); j++) {
            int i = possible_candidates[j];
            int coverage_score = inner_product(posting_list[i].begin(), posting_list[i].end(), coverage_tracker.begin(), 0);
            omp_set_lock(&sc_write_lock);
            if (coverage_score > max_score) {
                best_point = i;
                max_score = coverage_score;
            }
            omp_unset_lock(&sc_write_lock);
        }

        omp_destroy_lock(&sc_write_lock);

        if (best_point != -1) {
            solution.insert(best_point);
            vector<int> pl_best = posting_list[best_point];
            for (int i = 0; i < dataset_size; i++) {
                if (pl_best[i] == 1) {
                    if (coverage_tracker[i] != 0) {
                        coverage_tracker[i] -= 1;
                    }
                }
            }
        } else {
            cout << "Cannot find a point" << endl;
            break;
        }

    }
    return solution;
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

set<int> gfkc(string dataset, double coverage_factor, int distribution_req, int num_classes, int dataset_size) {
    // Reading posting list 
    // string posting_list_file = "/localdisk3/data-selection/data/metadata/" + dataset + "/" + to_string(coverage_factor) + "/resnet-18.txt";
    // cout << posting_list_file << endl;
    double pre_time = 0.0;
    double algo_time = 0.0;
    string posting_list_file = "/localdisk3/data-selection/data/metadata/cifar10/0.9/resnet-18_rf.txt";
    string labels_file = "/localdisk3/data-selection/data/metadata/cifar10/labels.txt";
    map<int, vector<int>> labels_dict;
    // initialize trackers
    /**
     * @todo coverage tracker sparse points
     * @todo fairness tracker class wise distribution req
    */
    vector<int> coverage_tracker(dataset_size, coverage_factor);
    vector<int> fairness_tracker(num_classes, distribution_req);
    set<int> coreset;
    set<int> delta;



    std::chrono::time_point<std::chrono::high_resolution_clock> pre_start, pre_end;
    std::chrono::duration<double> pre_elapsed;
    pre_start = std::chrono::high_resolution_clock::now();
    std::ifstream infile(posting_list_file);
    map<int, vector<int>> posting_list;
    if (infile.is_open()) {
        string line;
        int key = 0;
        while (getline(infile, line)) {
            stringstream ss(line);
            string value;
            vector<int> values(dataset_size, 0);
            while (ss >> value) {
                values[stoi(value)] = 1;
            }

            posting_list[key] = values;
            key += 1;
        }
        

    } else {
        cout << "Error opening input file: " << posting_list_file << endl;
    }
    infile.close();
    
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
    pre_end = std::chrono::high_resolution_clock::now();
    pre_elapsed = pre_end - pre_start;
    pre_time = pre_elapsed.count();
    cout << "Preprocessing Time: " << to_string(pre_time) << endl;

    
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

        omp_lock_t write_lock;
        omp_init_lock(&write_lock);

        # pragma omp parallel for
        for (int j = 0; j < possible_candidates.size(); j++) {
            int i = possible_candidates[j];
            int coverage_score = inner_product(posting_list[i].begin(), posting_list[i].end(), coverage_tracker.begin(), 0);
            int fairness_score = inner_product(labels_dict[i].begin(), labels_dict[i].end(), fairness_tracker.begin(), 0);
            omp_set_lock(&write_lock);
            if ((coverage_score + fairness_score) > max_score) {
                best_point = i;
                max_score = coverage_score + fairness_score;
            }
            omp_unset_lock(&write_lock);
        }
        omp_destroy_lock(&write_lock);

        if (best_point != -1) {
            coreset.insert(best_point);
            vector<int> pl_best = posting_list[best_point];
            for (int i = 0; i < dataset_size; i++) {
                if (pl_best[i] == 1) {
                    if (coverage_tracker[i] != 0) {
                        coverage_tracker[i] -= 1;
                    }
                }
            }
            vector<int> labels_best = labels_dict[best_point];
            for (int i = 0; i < num_classes; i++) {
                if (labels_best[i] == 1) {
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
    }
    algo_end = std::chrono::high_resolution_clock::now();
    algo_elapsed = algo_end - algo_start;
    algo_time = algo_elapsed.count();

    cout << "Algorithm time: " << to_string(algo_time) << endl;
    
    return coreset;
}





/**
 * ManyToMany Swapping Algorithm 
 * @param coverage_coreset coreset that satisfies K-coverage
 * @param dataset_name name of the dataset
 * @param num_classes number of classes/groups in the dataset
 * @param dataset_size size of the dataset
 * @param distribution_req fairness constraint for groups
 * @param coverage_factor number of representatives for each point in the coreset
*/
void many_to_many_swap(set<int> coverage_coreset, string dataset_name, int num_classes, int dataset_size, int distribution_req, int coverage_factor) {
    
    double pre_time = 0.0;
    double algo_time = 0.0;
    map<int, set<int>> labels_to_points;
    map<int, set<int>> coverage_coreset_distribution;
    set<int> g_left;
    set<int> g_extra;
    set<int> L;
    set<int> R;
    set<int> delta_minus_coverage_coreset;
    vector<int> coverage_tracker(dataset_size, 0);
    map<int, set<int>> posting_list = get_posting_list(dataset_size, dataset_name);

    
    std::chrono::time_point<std::chrono::high_resolution_clock> pre_start, pre_end;
    std::chrono::duration<double> pre_elapsed;
    pre_start = std::chrono::high_resolution_clock::now();
    for (int d = 0; d < dataset_size; d++) {
        set<int>::iterator it = coverage_coreset.find(d);
        if (it == coverage_coreset.end()) {
            delta_minus_coverage_coreset.insert(d);
        }
    }

    for (int i = 0; i < num_classes; i++) {
        set<int> points;
        labels_to_points[i] = points; 
    }
    string labels_file = "/localdisk3/data-selection/data/metadata/cifar10/labels.txt";
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
            labels_to_points[label].insert(id);
        }
    } else {
        cout << "Error opening input file: " << labels_file << endl;
    }

    pre_end = std::chrono::high_resolution_clock::now();
    pre_elapsed = pre_end - pre_start;
    pre_time = pre_elapsed.count();
    cout << "Preprocessing Time: " << to_string(pre_time) << endl;

    std::chrono::time_point<std::chrono::high_resolution_clock> algo_start, algo_end;
    std::chrono::duration<double> algo_elapsed;
    algo_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_classes; i++) {
        set<int> class_points;
        set_intersection(coverage_coreset.begin(), coverage_coreset.end(), labels_to_points[i].begin(), labels_to_points[i].end(), inserter(class_points, class_points.begin()));
        coverage_coreset_distribution[i] = class_points;
    }

    for (auto const& x : coverage_coreset_distribution) {
        int group_id = x.first;
        if (x.second.size() < distribution_req) {
            g_left.insert(group_id);
        } 
        
        if (x.second.size() > distribution_req) {
            g_extra.insert(group_id);
            int number_of_points = x.second.size() - distribution_req;
            std::sample(x.second.begin(), x.second.end(), inserter(L, L.begin()), number_of_points, std::mt19937 {std::random_device{} ()});
        }
    }

    /**
     * @todo what if they're aren't any points you can swap based on the number of groups in g_extra. In this case we just random sample for each group that was left
    */

    if (L.size() > 0) {
        for (auto gl : g_left) {
            set_intersection(delta_minus_coverage_coreset.begin(), delta_minus_coverage_coreset.end(), labels_to_points[gl].begin(), labels_to_points[gl].end(), inserter(R, R.begin()));
        }

        for (auto l : L) {
            coverage_coreset.erase(l);
            // calculate affected points
            for (auto x : posting_list[l]) {
                set<int> representatives;
                set_intersection(coverage_coreset.begin(), coverage_coreset.end(), posting_list[x].begin(), posting_list[x].end(), inserter(representatives, representatives.begin()));
                if (representatives.size() < coverage_factor) {
                    for (auto r : representatives) {
                        coverage_tracker[r] += 1;
                    }
                }
                representatives.clear();
            }
        }

        set<int> r_star = set_cover(coverage_tracker, R, dataset_size, dataset_name);

        for (auto r : r_star) {
            coverage_coreset.insert(r);
        }


        
    } else {
        // we can't swap so we add randomly 
        for (auto gl : g_left) {
            int number_of_points = distribution_req - coverage_coreset_distribution[gl].size();
            set<int> R_temp;
            set_intersection(delta_minus_coverage_coreset.begin(), delta_minus_coverage_coreset.end(), labels_to_points[gl].begin(), labels_to_points[gl].end(), inserter(R_temp, R_temp.begin()));
            std::sample(R_temp.begin(), R_temp.end(), inserter(coverage_coreset, coverage_coreset.begin()), number_of_points, std::mt19937 {std::random_device{} ()});
        }
    }

    algo_end = std::chrono::high_resolution_clock::now();
    algo_elapsed = algo_end - algo_start;
    algo_time = algo_elapsed.count();

    cout << "Algorithm time: " << to_string(algo_time) << endl;
    cout << "Solution Size: " << coverage_coreset.size() << endl;
    
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
    cout << "Coverage Factor: " << coverage_factor << endl;
    cout << "Distribution Req: " << distribution_req << endl;

    set<int> coverage_coreset = gfkc(dataset, coverage_factor, 0, num_classes, dataset_size);
    many_to_many_swap(coverage_coreset, dataset, num_classes, dataset_size, distribution_req, coverage_factor);
    return 0;
}