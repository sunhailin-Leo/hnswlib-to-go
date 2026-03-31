package hnswgo

/*
#cgo CXXFLAGS: -std=c++11
#cgo !windows LDFLAGS: -L${SRCDIR} -lhnsw -lm -lstdc++
#cgo windows LDFLAGS: -L${SRCDIR} -lhnsw -lstdc++
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "hnsw_wrapper.h"
*/
import "C"
import (
	"math"
	"runtime"
	"sync"
	"unsafe"
)

// SpaceType defines the distance metric used by the HNSW index.
const (
	SpaceIP     = "ip"
	SpaceCosine = "cosine"
	SpaceL2     = "l2"
)

// HNSW represents an HNSW index instance.
type HNSW struct {
	index     C.HNSW
	spaceType string
	dim       int
	normalize bool
}

// spaceChar returns the C char representation for the space type.
func spaceChar(spaceType string) C.char {
	if spaceType == SpaceIP || spaceType == SpaceCosine {
		return C.char('i')
	}
	return C.char('l')
}

// New creates a new HNSW index.
func New(dim, M, efConstruction, randSeed int, maxElements uint32, spaceType string) *HNSW {
	hnsw := &HNSW{
		dim:       dim,
		spaceType: spaceType,
		normalize: spaceType == SpaceCosine,
	}
	hnsw.index = C.initHNSW(
		C.int(dim), C.uint64_t(maxElements), C.int(M), C.int(efConstruction),
		C.int(randSeed), spaceChar(spaceType), C.bool(false))
	return hnsw
}

// NewWithReplaceDeleted creates a new HNSW index with replace-deleted support enabled.
func NewWithReplaceDeleted(dim, M, efConstruction, randSeed int, maxElements uint32, spaceType string) *HNSW {
	hnsw := &HNSW{
		dim:       dim,
		spaceType: spaceType,
		normalize: spaceType == SpaceCosine,
	}
	hnsw.index = C.initHNSW(
		C.int(dim), C.uint64_t(maxElements), C.int(M), C.int(efConstruction),
		C.int(randSeed), spaceChar(spaceType), C.bool(true))
	return hnsw
}

// Load loads an HNSW index from a file.
func Load(location string, dim int, spaceType string) *HNSW {
	hnsw := &HNSW{
		dim:       dim,
		spaceType: spaceType,
		normalize: spaceType == SpaceCosine,
	}
	pLocation := C.CString(location)
	defer C.free(unsafe.Pointer(pLocation))
	hnsw.index = C.loadHNSW(pLocation, C.int(dim), spaceChar(spaceType))
	return hnsw
}

// Free releases the HNSW index memory. Returns true if the index was freed.
func (h *HNSW) Free() bool {
	if h.index == nil {
		return false
	}
	C.freeHNSW(h.index, spaceChar(h.spaceType))
	h.index = nil
	runtime.GC()
	return true
}

// Unload is an alias for Free for backward compatibility.
// Deprecated: Use Free instead.
func (h *HNSW) Unload() bool {
	return h.Free()
}

// Save persists the HNSW index to a file.
func (h *HNSW) Save(location string) bool {
	if h.index == nil {
		return false
	}
	pLocation := C.CString(location)
	defer C.free(unsafe.Pointer(pLocation))
	C.saveHNSW(h.index, pLocation)
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

// AddPoint adds a point to the index.
func (h *HNSW) AddPoint(vector []float32, label uint32) bool {
	if h.index == nil {
		return false
	}
	if h.normalize {
		vector = normalizeVector(vector)
	}
	C.addPoint(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.uint64_t(label), C.bool(false))
	return true
}

// AddPointWithReplace adds a point to the index, replacing a previously deleted element if available.
// Only works if the index was created with NewWithReplaceDeleted.
func (h *HNSW) AddPointWithReplace(vector []float32, label uint32) bool {
	if h.index == nil {
		return false
	}
	if h.normalize {
		vector = normalizeVector(vector)
	}
	C.addPoint(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.uint64_t(label), C.bool(true))
	return true
}

// AddBatchPoints add some points on graph with goroutine
func (h *HNSW) AddBatchPoints(vectors [][]float32, labels []uint32, coroutines int) bool {
	if len(vectors) != len(labels) || coroutines < 1 {
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
	Clabel := make([]C.uint64_t, N, N)
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

// SearchBatchKNN search multiple points on graph with knn-algorithm
func (h *HNSW) SearchBatchKNN(vectors [][]float32, N, coroutines int) ([][]uint32, [][]float32) {
	if coroutines < 1 {
		coroutines = 1
	}

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

// ResizeIndex resizes the index to accommodate a new maximum number of elements.
func (h *HNSW) ResizeIndex(newMaxElements uint32) bool {
	if h.index == nil {
		return false
	}
	return bool(C.resizeIndex(h.index, C.uint64_t(newMaxElements)))
}

// MarkDelete marks a label as deleted (soft delete).
func (h *HNSW) MarkDelete(label uint32) bool {
	if h.index == nil {
		return false
	}
	return bool(C.markDelete(h.index, C.uint64_t(label)))
}

// UnmarkDelete removes the deleted mark from a label.
func (h *HNSW) UnmarkDelete(label uint32) bool {
	if h.index == nil {
		return false
	}
	return bool(C.unmarkDelete(h.index, C.uint64_t(label)))
}

// GetLabelIsMarkedDeleted checks if a label is marked as deleted.
func (h *HNSW) GetLabelIsMarkedDeleted(label uint32) bool {
	if h.index == nil {
		return false
	}
	return bool(C.isMarkedDeleted(h.index, C.uint64_t(label)))
}

// UpdatePoint updates the vector for an existing point in the index.
func (h *HNSW) UpdatePoint(vector []float32, label uint32, updateNeighborProbability float32) bool {
	if h.index == nil || len(vector) == 0 {
		return false
	}
	return bool(C.updatePoint(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.uint64_t(label), C.float(updateNeighborProbability)))
}

// UpdateBatchPoints updates multiple points in the index concurrently.
func (h *HNSW) UpdateBatchPoints(vectors [][]float32, labels []uint32, updateNeighborProbabilities []float32, coroutines int) bool {
	if h.index == nil || len(vectors) != len(labels) || len(labels) != len(updateNeighborProbabilities) || coroutines < 1 {
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
		go func(thisVectors [][]float32, thisLabels []uint32, thisProb []float32) {
			defer wg.Done()
			for j := 0; j < len(thisVectors); j++ {
				h.UpdatePoint(thisVectors[j], thisLabels[j], thisProb[j])
			}
		}(vectors[i*b:end], labels[i*b:end], updateNeighborProbabilities[i*b:end])
	}

	wg.Wait()
	return true
}

// GetMaxElements returns the maximum number of elements the index can hold.
func (h *HNSW) GetMaxElements() int {
	if h.index == nil {
		return 0
	}
	return int(C.getMaxElements(h.index))
}

// GetCurrentElementCount returns the current number of elements in the index.
func (h *HNSW) GetCurrentElementCount() int {
	if h.index == nil {
		return 0
	}
	return int(C.getCurrentElementCount(h.index))
}

// GetDeleteCount returns the number of elements marked as deleted.
func (h *HNSW) GetDeleteCount() int {
	if h.index == nil {
		return 0
	}
	return int(C.getDeleteCount(h.index))
}

// GetVectorByLabel retrieves the vector associated with the given label.
// Returns nil if the label is not found, has been deleted, or the index is nil.
func (h *HNSW) GetVectorByLabel(label uint32) []float32 {
	if h.index == nil {
		return nil
	}
	outData := make([]float32, h.dim)
	result := int(C.getDataByLabel(h.index, C.uint64_t(label), (*C.float)(unsafe.Pointer(&outData[0])), C.int(h.dim)))
	if result < 0 {
		return nil
	}
	return outData
}
