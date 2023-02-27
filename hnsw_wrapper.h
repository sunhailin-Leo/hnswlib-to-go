// hnsw_wrapper.h
#ifdef __cplusplus
extern "C" {
#endif
typedef void *HNSW;

HNSW initHNSW(int dim, unsigned long int max_elements, int M, int ef_construction, int rand_seed, char stype);

HNSW loadHNSW(char *location, int dim, char stype);

HNSW saveHNSW(HNSW index, char *location);

void addPoint(HNSW index, float *vec, unsigned long int label);

int searchKnn(HNSW index, float *vec, int N, unsigned long int *label, float *dist);

void setEf(HNSW index, int ef);

bool resizeIndex(HNSW index, unsigned long int new_max_elements);

bool markDelete(HNSW index, unsigned long int label);

bool unmarkDelete(HNSW index, unsigned long int label);

bool isMarkedDeleted(HNSW index, unsigned long int label);

bool updatePoint(HNSW index, float *vec, unsigned long int label);

int getMaxElements(HNSW index);

int getCurrentElementCount(HNSW index);

int getDeleteCount(HNSW index);

void getDataByLabel(HNSW index, unsigned long int label, float* out_data);
#ifdef __cplusplus
}
#endif