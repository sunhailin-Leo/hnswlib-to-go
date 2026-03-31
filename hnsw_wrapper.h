// hnsw_wrapper.h
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef void *HNSW;

HNSW initHNSW(int dim, uint64_t max_elements, int M, int ef_construction, int rand_seed, char stype, bool allow_replace_deleted);

HNSW loadHNSW(char *location, int dim, char stype);

void freeHNSW(HNSW index, char stype);

void saveHNSW(HNSW index, char *location);

void addPoint(HNSW index, float *vec, uint64_t label, bool replace_deleted);

int searchKnn(HNSW index, float *vec, int N, uint64_t *label, float *dist);

void setEf(HNSW index, int ef);

bool resizeIndex(HNSW index, uint64_t new_max_elements);

bool markDelete(HNSW index, uint64_t label);

bool unmarkDelete(HNSW index, uint64_t label);

bool isMarkedDeleted(HNSW index, uint64_t label);

bool updatePoint(HNSW index, float *vec, uint64_t label, float updateNeighborProbability);

int getMaxElements(HNSW index);

int getCurrentElementCount(HNSW index);

int getDeleteCount(HNSW index);

int getDataByLabel(HNSW index, uint64_t label, float* out_data, int dim);

#ifdef __cplusplus
}
#endif