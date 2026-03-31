// hnsw_wrapper.cc
#include <vector>
#include <iostream>
#include "hnswlib/hnswlib.h"
#include "hnsw_wrapper.h"
#include <thread>
#include <atomic>

HNSW initHNSW(int dim, uint64_t max_elements, int M, int ef_construction, int rand_seed, char stype, bool allow_replace_deleted) {
    hnswlib::SpaceInterface<float> *space;
    if (stype == 'i') {
        space = new hnswlib::InnerProductSpace(dim);
    } else {
        space = new hnswlib::L2Space(dim);
    }
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(
        space, (size_t)max_elements, M, ef_construction, rand_seed, allow_replace_deleted);
    return (void *) appr_alg;
}

HNSW loadHNSW(char *location, int dim, char stype) {
    hnswlib::SpaceInterface<float> *space;
    if (stype == 'i') {
        space = new hnswlib::InnerProductSpace(dim);
    } else {
        space = new hnswlib::L2Space(dim);
    }
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(
        space, std::string(location), false, 0);
    return (void *) appr_alg;
}

void freeHNSW(HNSW index, char stype) {
    if (index == nullptr) return;
    delete (hnswlib::HierarchicalNSW<float> *) index;
}

void saveHNSW(HNSW index, char *location) {
    ((hnswlib::HierarchicalNSW<float> *) index)->saveIndex(location);
}

void addPoint(HNSW index, float *vec, uint64_t label, bool replace_deleted) {
    ((hnswlib::HierarchicalNSW<float> *) index)->addPoint(vec, (hnswlib::labeltype)label, replace_deleted);
}

int searchKnn(HNSW index, float *vec, int N, uint64_t *label, float *dist) {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> gt;
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
        *(label + i) = (uint64_t)pair.second;
        gt.pop();
    }
    return n;
}

void setEf(HNSW index, int ef) {
    ((hnswlib::HierarchicalNSW<float> *) index)->setEf(ef);
}

bool resizeIndex(HNSW index, uint64_t new_max_elements) {
    try {
        ((hnswlib::HierarchicalNSW<float> *) index)->resizeIndex((size_t)new_max_elements);
    } catch (const std::exception &e) {
        return false;
    }
    return true;
}

bool markDelete(HNSW index, uint64_t label) {
    try {
        ((hnswlib::HierarchicalNSW<float> *) index)->markDelete((hnswlib::labeltype)label);
        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

bool unmarkDelete(HNSW index, uint64_t label) {
    try {
        ((hnswlib::HierarchicalNSW<float> *) index)->unmarkDelete((hnswlib::labeltype)label);
        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

bool isMarkedDeleted(HNSW index, uint64_t label) {
    std::unique_lock<std::mutex> lock_table(((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_lock);
    auto search = ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.find((hnswlib::labeltype)label);

    if (search != ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.end()) {
        bool res = ((hnswlib::HierarchicalNSW<float> *) index)->isMarkedDeleted(search->second);
        lock_table.unlock();
        return res;
    }
    return false;
}

bool updatePoint(HNSW index, float *vec, uint64_t label, float updateNeighborProbability) {
    std::unique_lock<std::mutex> lock_table(((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_lock);
    auto search = ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.find((hnswlib::labeltype)label);

    if (search != ((hnswlib::HierarchicalNSW<float> *) index)->label_lookup_.end()) {
        hnswlib::tableint existingInternalId = search->second;
        lock_table.unlock();
        ((hnswlib::HierarchicalNSW<float> *) index)->updatePoint(vec, existingInternalId, updateNeighborProbability);
        return true;
    }
    return false;
}

int getDataByLabel(HNSW index, uint64_t label, float* out_data, int dim) {
    try {
        auto data = ((hnswlib::HierarchicalNSW<float>*)index)->getDataByLabel<float>((hnswlib::labeltype)label);
        size_t size = data.size();
        for (size_t i = 0; i < size; i++) {
            out_data[i] = data[i];
        }
        return (int)size;
    } catch (const std::exception &e) {
        return -1;
    }
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

