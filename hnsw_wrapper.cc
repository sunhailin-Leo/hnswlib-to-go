//hnsw_wrapper.cpp
#include <vector>
#include <iostream>
#include "hnswlib/hnswlib.h"
#include "hnsw_wrapper.h"
#include <thread>
#include <atomic>

HNSW initHNSW(int dim, unsigned long int max_elements, int M, int ef_construction, int rand_seed, char stype) {
    hnswlib::SpaceInterface<float> *space;
    if (stype == 'i') {
        space = new hnswlib::InnerProductSpace(dim);
    } else {
        space = new hnswlib::L2Space(dim);
    }
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, max_elements, M,
                                                                                    ef_construction, rand_seed);
    return (void *) appr_alg;
}

HNSW loadHNSW(char *location, int dim, char stype) {
    hnswlib::SpaceInterface<float> *space;
    if (stype == 'i') {
        space = new hnswlib::InnerProductSpace(dim);
    } else {
        space = new hnswlib::L2Space(dim);
    }
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, std::string(location), false,
                                                                                    0);
    return (void *) appr_alg;
}

HNSW saveHNSW(HNSW index, char *location) {
    ((hnswlib::HierarchicalNSW<float> *) index)->saveIndex(location);
    return 0;
}

void addPoint(HNSW index, float *vec, unsigned long int label) {
    ((hnswlib::HierarchicalNSW<float> *) index)->addPoint(vec, label);
}

int searchKnn(HNSW index, float *vec, int N, unsigned long int *label, float *dist) {
    std::priority_queue <std::pair<float, hnswlib::labeltype>> gt;
    try {
        gt = ((hnswlib::HierarchicalNSW<float> *) index)->searchKnn(vec, N);
    } catch (const std::exception &e) {
        return 0;
    }

    int n = gt.size();
    std::pair<float, hnswlib::labeltype> pair;
    for (int i = n - 1; i >= 0; i--) {
        pair = gt.top();
        *(dist + i) = pair.first;
        *(label + i) = pair.second;
        gt.pop();
    }
    return n;
}

void setEf(HNSW index, int ef) {
    ((hnswlib::HierarchicalNSW<float> *) index)->ef_ = ef;
}

bool resizeIndex(HNSW index, unsigned long int new_max_elements) {
    if (new_max_elements < ((hnswlib::HierarchicalNSW<float> *) index)->getCurrentElementCount()) {
        return false;
    }
    try {
        ((hnswlib::HierarchicalNSW<float> *) index)->resizeIndex(new_max_elements);
    } catch (const std::exception &e) {
        return false;
    }
    return true;
}

bool markDelete(HNSW index, unsigned long int label) {
    try {
        ((hnswlib::HierarchicalNSW<float> *) index)->markDelete(label);
        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

bool unmarkDelete(HNSW index, unsigned long int label) {
    try {
        ((hnswlib::HierarchicalNSW<float> *) index)->unmarkDelete(label);
        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

bool isMarkedDeleted(HNSW index, unsigned long int label) {
    std::unique_lock <std::mutex> lock_table(((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_lock);
    auto search = ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.find(label);

    if (search != ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.end()) {
        bool res = ((hnswlib::HierarchicalNSW<float> *) index)->isMarkedDeleted(search->second);
        lock_table.unlock();
        return res;
    }
    return false;
}

bool updatePoint(HNSW index, float *vec, unsigned long int label) {
    std::unique_lock <std::mutex> lock_table(((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_lock);
    auto search = ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.find(label);

    if (search != ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.end()) {
        hnswlib::tableint existingInternalId = search->second;
        lock_table.unlock();
        // const void *dataPoint, tableint internalId, float updateNeighborProbability
        ((hnswlib::HierarchicalNSW<float> *) index)->updatePoint(vec, existingInternalId, 1.0);
        return true;
    }
    return false;
}

void getDataByLabel(HNSW index, unsigned long int label, float* out_data) {
    auto data = ((hnswlib::HierarchicalNSW<float>*)index)->getDataByLabel<float>(label);
    std::vector<float>* vec = new std::vector<float>(data.begin(), data.end());
    if (vec == nullptr) {
        return;
    }

    size_t size = vec->size();
    for (size_t i = 0; i < size; i++) {
        out_data[i] = (*vec)[i];
    }

    delete vec;
}

int getMaxElements(HNSW index) {
    return ((hnswlib::HierarchicalNSW<float> *) index)->getMaxElements();
}

int getCurrentElementCount(HNSW index) {
    return ((hnswlib::HierarchicalNSW<float> *) index)->getCurrentElementCount();
}

int getDeleteCount(HNSW index) {
    return ((hnswlib::HierarchicalNSW<float> *) index)->getDeletedCount();
}

