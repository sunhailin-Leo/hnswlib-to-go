package hnswgo

import "C"

/*
#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS: -L${SRCDIR} -lhnsw -lm
#include <stdlib.h>
#include <stdbool.h>
#include "hnsw_wrapper.h"

HNSW initHNSW(int dim, unsigned long int max_elements, int M, int ef_construction, int rand_seed, char stype);
HNSW loadHNSW(char *location, int dim, char stype);
void addPoint(HNSW index, float *vec, unsigned long int label);
int searchKnn(HNSW index, float *vec, int N, unsigned long int *label, float *dist);
void setEf(HNSW index, int ef);
bool resizeIndex(HNSW index, unsigned long int new_max_elements);
bool markDelete(HNSW index, unsigned long int label);
bool unmarkDelete(HNSW index, unsigned long int label);
bool isMarkedDeleted(HNSW index, unsigned long int label);
bool updatePoint(HNSW index, float *vec, unsigned long int label);

void getDataByLabel(HNSW index, unsigned long int label, float* out_data);
*/
import "C"
import (
	"math"
	"runtime"
	"sync"
	"unsafe"
)

func toSlice(v *C.float, len int) []float32 {
	// 创建一个指向C数组的slice
	slice := (*[1 << 30]float32)(unsafe.Pointer(v))[:len:len]
	// 复制slice的值，将其转换为一个新的Go切片
	return append([]float32(nil), slice...)
}

type HNSW struct {
	index     C.HNSW
	spaceType string
	dim       int
	normalize bool
}

// New make a hnsw graph
func New(dim, M, efConstruction, randSeed int, maxElements uint32, spaceType string) *HNSW {
	var hnsw HNSW
	hnsw.dim = dim
	hnsw.spaceType = spaceType
	if spaceType == "ip" {
		hnsw.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.char('i'))
	} else if spaceType == "cosine" {
		hnsw.normalize = true
		hnsw.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.char('i'))
	} else {
		hnsw.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.char('l'))
	}
	return &hnsw
}

// Load load a hnsw graph
func Load(location string, dim int, spaceType string) *HNSW {
	var hnsw HNSW
	hnsw.dim = dim
	hnsw.spaceType = spaceType

	pLocation := C.CString(location)
	if spaceType == "ip" {
		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('i'))
	} else if spaceType == "cosine" {
		hnsw.normalize = true
		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('i'))
	} else {
		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('l'))
	}
	C.free(unsafe.Pointer(pLocation))
	return &hnsw
}

// Unload TODO Test for release the graph memory
func (h *HNSW) Unload() bool {
	if h.index == nil {
		return false
	}
	C.free(unsafe.Pointer(h.index))
	h.index = nil
	// Free memory ASAP, but need to check the memory usage
	runtime.GC()
	return true
}

// Save save graph node on graph
func (h *HNSW) Save(location string) bool {
	if h.index == nil {
		return false
	}
	pLocation := C.CString(location)
	C.saveHNSW(h.index, pLocation)
	C.free(unsafe.Pointer(pLocation))
	return true
}

// normalizeVector normalize vector
func normalizeVector(vector []float32) []float32 {
	var norm float32
	for i := 0; i < len(vector); i++ {
		norm += vector[i] * vector[i]
	}
	norm = 1.0 / (float32(math.Sqrt(float64(norm))) + 1e-15)
	for i := 0; i < len(vector); i++ {
		vector[i] = vector[i] * norm
	}
	return vector
}

// AddPoint add a point on graph
func (h *HNSW) AddPoint(vector []float32, label uint32) bool {
	if h.index == nil {
		return false
	}
	if h.normalize {
		vector = normalizeVector(vector)
	}
	C.addPoint(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.ulong(label))
	return true
}

// AddBatchPoints add some points on graph with goroutine
func (h *HNSW) AddBatchPoints(vectors [][]float32, labels []uint32, coroutines int) bool {
	if len(vectors) != len(labels) {
		return false
	}

	b := len(vectors) / coroutines
	var wg sync.WaitGroup
	for i := 0; i < coroutines; i++ {
		wg.Add(1)

		end := (i + 1) * b
		if i == coroutines-1 && len(vectors) > end {
			end = len(vectors)
		}
		go func(thisVectors [][]float32, thisLabels []uint32) {
			defer wg.Done()
			for j := 0; j < len(thisVectors); j++ {
				h.AddPoint(thisVectors[j], thisLabels[j])
			}
		}(vectors[i*b:end], labels[i*b:end])
	}

	wg.Wait()
	return true
}

// SearchKNN search points on graph with knn-algorithm
func (h *HNSW) SearchKNN(vector []float32, N int) ([]uint32, []float32) {
	if h.index == nil {
		return nil, nil
	}
	Clabel := make([]C.ulong, N, N)
	Cdist := make([]C.float, N, N)
	if h.normalize {
		vector = normalizeVector(vector)
	}
	numResult := int(C.searchKnn(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.int(N), &Clabel[0], &Cdist[0]))
	labels := make([]uint32, N)
	dists := make([]float32, N)
	for i := 0; i < numResult; i++ {
		labels[i] = uint32(Clabel[i])
		dists[i] = float32(Cdist[i])
	}
	return labels[:numResult], dists[:numResult]
}

func (h *HNSW) SearchBatchKNN(vectors [][]float32, N, coroutines int) ([][]uint32, [][]float32) {
	var lock sync.Mutex
	labelList := make([][]uint32, len(vectors))
	distList := make([][]float32, len(vectors))

	b := len(vectors) / coroutines
	var wg sync.WaitGroup
	for i := 0; i < coroutines; i++ {
		wg.Add(1)

		end := (i + 1) * b
		if i == coroutines-1 && len(vectors) > end {
			end = len(vectors)
		}
		go func(i int) {
			defer wg.Done()
			for j := i * b; j < end; j++ {
				labels, dist := h.SearchKNN(vectors[j], N)
				lock.Lock()
				labelList[j] = labels
				distList[j] = dist
				lock.Unlock()
			}
		}(i)
	}
	wg.Wait()
	return labelList, distList
}

// SetEf set ef argument on graph
func (h *HNSW) SetEf(ef int) {
	if h.index == nil {
		return
	}
	C.setEf(h.index, C.int(ef))
}

// SetNormalize set normalize on graph
func (h *HNSW) SetNormalize(isNeedNormalize bool) {
	h.normalize = isNeedNormalize
}

// ResizeIndex set new elements count to resize index
func (h *HNSW) ResizeIndex(newMaxElements uint32) bool {
	isResize := bool(C.resizeIndex(h.index, C.ulong(newMaxElements)))
	return isResize
}

// MarkDelete mark a label to delete mode
func (h *HNSW) MarkDelete(label uint32) bool {
	isMark := bool(C.markDelete(h.index, C.ulong(label)))
	return isMark
}

// UnmarkDelete unmark a label to delete mode
func (h *HNSW) UnmarkDelete(label uint32) bool {
	isUnmark := bool(C.unmarkDelete(h.index, C.ulong(label)))
	return isUnmark
}

// GetLabelIsMarkedDeleted get label isDelete
func (h *HNSW) GetLabelIsMarkedDeleted(label uint32) bool {
	isDelete := bool(C.isMarkedDeleted(h.index, C.ulong(label)))
	return isDelete
}

// GetMaxElements get index max elements
func (h *HNSW) GetMaxElements() int {
	maxElements := int(C.getMaxElements(h.index))
	return maxElements
}

// GetCurrentElementCount get index current elements
func (h *HNSW) GetCurrentElementCount() int {
	elementCnt := int(C.getCurrentElementCount(h.index))
	return elementCnt
}

// GetDeleteCount get index count which mark deleted
func (h *HNSW) GetDeleteCount() int {
	deleteElementCnt := int(C.getDeleteCount(h.index))
	return deleteElementCnt
}

// GetVectorByLabel get index by label
func (h *HNSW) GetVectorByLabel(label uint32, dim int) []float32 {
	var outDataPtr C.float
	C.getDataByLabel(h.index, C.ulong(label), &outDataPtr)
	outData := make([]float32, dim)
	for i := 0; i < dim; i++ {
		outData[i] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(&outDataPtr)) + uintptr(i)*unsafe.Sizeof(C.float(0)))))
	}
	return outData
}
